from collections import OrderedDict
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import repeat

from models.MoE.softmoe import SoftMoELayerWrapper



def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth per sample when applied to residual paths."""

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample for residual blocks."""

    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """2D image to patch embedding."""

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class TaskAttention(nn.Module):
    """Task-specific attention that routes each task token to top-k patch tokens."""

    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 num_tasks=1, capacity_factor=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qs = nn.ModuleList([nn.Linear(dim, dim, bias=qkv_bias) for _ in range(num_tasks)])
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.num_tasks = num_tasks
        self.capacity_factor = capacity_factor

        self.in_features = dim
        self.experts = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.num_tasks)])

    def forward(self, x: torch.Tensor, heads=None) -> torch.Tensor:
        B, N, C = x.shape  # x: [B, num_tasks + tokens, C]
        feature = x[:, self.num_tasks :, :]  # feature: [B, tokens, C]
        feature_multihead = feature.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B, H, tokens, Dh]
        kv = self.kv(feature).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, B, H, tokens, Dh]
        k, v = kv[0], kv[1]  # k,v: [B, H, tokens, Dh]

        # Limit attention to the top-k tokens per task to reduce compute.
        top_k = max(1, min(self.capacity_factor, feature.size(1)))

        qs = []  # will become [B, H, num_tasks, 1, Dh]
        for i in range(self.num_tasks):
            tasktoken = x[:, i, :].unsqueeze(1)
            q = self.qs[i](tasktoken).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            qs.append(q)

        qs = torch.cat(qs, dim=2)  # [B, H, num_tasks, 1, Dh]

        atten_tf = (qs @ k.transpose(-2, -1)) * self.scale  # [B, H, num_tasks, tokens]
        atten_tf = self.attn_drop(atten_tf)

        top_k_value, top_k_indice = torch.topk(atten_tf, k=top_k, dim=-1, sorted=False)  # both [B, H, num_tasks, top_k]
        top_k_value_softmax = top_k_value.softmax(dim=-1)

        atten_topk = torch.zeros_like(atten_tf)  # [B, H, num_tasks, tokens]
        atten_topk.scatter_(-1, top_k_indice, top_k_value_softmax)

        attn_token = (atten_topk @ v).transpose(1, 2).reshape(B, -1, C)  # [B, num_tasks, C]

        feature_multihead_expand = feature_multihead.unsqueeze(2).expand(-1, -1, self.num_tasks, -1, -1)  # [B, H, num_tasks, tokens, Dh]
        top_k_indice_expand = top_k_indice.unsqueeze(-1).expand(-1, -1, -1, -1, v.size(-1))  # [B, H, num_tasks, top_k, Dh]
        feature_topk = torch.gather(feature_multihead_expand, -2, top_k_indice_expand)  # [B, H, num_tasks, top_k, Dh]

        feature_topk *= top_k_value_softmax.unsqueeze(-1).expand_as(feature_topk)

        feature_pad = torch.zeros_like(feature_multihead_expand)  # [B, H, num_tasks, tokens, Dh]
        feature_pad.scatter_(-2, top_k_indice_expand, feature_topk)
        feature_pad = feature_pad.permute(0, 2, 3, 1, 4).reshape(B, self.num_tasks, -1, C)  # [B, num_tasks, tokens, C]

        feature_output = torch.stack(
            [f_i(feature_pad[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )
        token_output = torch.stack(
            [f_i(attn_token[:, i, :]) for i, f_i in enumerate(self.experts)], dim=1
        )
        task_topk_weight = torch.ones_like(feature_output)

        feature_output = torch.einsum("btnc,btnc->bnc", feature_output, task_topk_weight)
        output = torch.cat((token_output, feature_output), dim=1)
        return output


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 num_tasks=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.num_tasks = num_tasks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # -------------------------------------------------------------------
        mask = torch.ones_like(attn, requires_grad=False)
        if self.num_tasks > 1:
            for i in range(self.num_tasks - 1):
                for j in range(i + 1, self.num_tasks):
                    mask[:, :, i, j] = mask[:, :, j, i] = 0

        attn1 = attn * mask
        x = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in ViT / MLP-Mixer."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class ExpChoiceBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path_ratio=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_experts=8, num_tasks=3,
                 capacity_factor=256):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = TaskAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop,
            proj_drop_ratio=drop,
            num_tasks=num_tasks,
            capacity_factor=capacity_factor,
        )
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_tasks = num_tasks

        # SoftMoE routes tokens to experts in the feed-forward block.
        self.mlp = SoftMoELayerWrapper(
            dim,
            num_experts=num_experts,
            slots_per_expert=1,
            layer=partial(Mlp, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop),
            normalize=True,
        )

        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, heads=None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), heads))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class MultiTask_Agg(nn.Module):
    """Multi-task aggregation transformer with task tokens and MoE MLPs."""

    def __init__(self, patch_size=16, in_c=3, classes=None,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.0,
                 attn_drop_ratio=0.0, drop_path_ratio=0.0, norm_layer=None,
                 act_layer=None, num_tasks=1, num_experts=16, moe_gate_dim=770, capacity_factors=None):

        super().__init__()
        if classes is None:
            classes = [2, 2, 2]
        if isinstance(classes, int):
            classes = [classes] * num_tasks

        self.classes = classes
        embed_dim_in = embed_dim
        self.embed_dim = 768  # embed_dim for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.depth = depth
        self.norm_layer = norm_layer
        self.num_experts = num_experts

        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_ratio
        self.attn_drop_rate = attn_drop_ratio
        self.num_tasks = num_tasks
        self.moe_gate_dim = moe_gate_dim

        blocks = []
        capacity_factors = capacity_factors or [256] * depth
        if len(capacity_factors) != depth:
            raise ValueError(f"capacity_factors length ({len(capacity_factors)}) must equal depth ({depth})")

        for i in range(self.depth):
            blocks.append(
                ExpChoiceBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    num_experts=num_experts,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path_ratio=drop_path_ratio,
                    norm_layer=self.norm_layer,
                    num_tasks=num_tasks,
                    capacity_factor=capacity_factors[i],
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.layer1 = nn.Sequential(
            nn.Linear(embed_dim_in, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )

        self.norm = norm_layer(self.embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.embed_dim = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.classes[1]) if classes > 0 else nn.Identity()

        self.task_tokens = nn.Parameter(torch.zeros(1, self.num_tasks, self.embed_dim))

        self.heads = nn.ModuleList([nn.Linear(self.embed_dim, classes[i]) for i in range(self.num_tasks)])

        self.pos_layer = PPEG(dim=self.embed_dim)

        # Weight init
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.task_tokens, std=0.02)
        self.apply(_init_vit_weights)


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)

        H = x.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            # Pad to a square sequence so the depth-wise convolutions see a grid.
            x = torch.cat([x, x[:, :add_length, :]], dim=1)
        x = self.pos_layer(x, _H, _W)

        task_tokens = repeat(self.task_tokens, '() n d -> b n d', b=x.size(0))
        if self.dist_token is None:
            x = torch.cat((task_tokens, x), dim=1)
        else:
            x = torch.cat((task_tokens, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        return x


    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        x = self.forward_features(x)
        x = self.norm(x)
        res: Dict[int, torch.Tensor] = {}
        for i in range(self.num_tasks):
            res[int(i)] = self.heads[i](x[:, i])
        return res


class PPEG(nn.Module):
    """Depth-wise convolutional positional encoding."""

    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


def masked_softmax(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = tensor != 0
    tensor_exp = torch.exp(tensor) * mask
    tensor_exp_sum = torch.sum(tensor_exp, dim=dim, keepdim=True)
    tensor_exp_sum = tensor_exp_sum + (tensor_exp_sum == 0).float()
    softmax = tensor_exp / tensor_exp_sum

    softmax = softmax * mask
    return softmax

if __name__ == '__main__':
    model = MultiTask_Agg(
        patch_size=16,
        embed_dim=512,
        depth=4,
        num_heads=2,
        representation_size=None,
        classes=2,
        num_tasks=2,
        capacity_factors=[4, 4, 4, 4],
    ).cuda()

    a = torch.FloatTensor(4, 256, 512).cuda()
    out = model(a)
    print(out[0].size())
    print(model.heads)