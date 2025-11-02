from functools import partial
from collections import OrderedDict
import torch.nn as nn
from einops import repeat
from torch.nn import functional as F
from collections import OrderedDict 
import torchvision.models as models  
import numpy as np
from collections import Counter
import torch
from models.MoE.softmoe import SoftMoELayerWrapper
softmax = nn.Softmax(dim=1)



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
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

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class TaskAttention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 num_tasks=1,capacity_factor=2):
        super(TaskAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qs = nn.ModuleList([nn.Linear(dim, dim, bias=qkv_bias) for i in range(num_tasks) ])
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.num_tasks = self.num_tasks = num_tasks
        self.capacity_factor = capacity_factor
        
        self.in_features = dim
        self.experts = nn.ModuleList([nn.Linear(dim, dim,bias=False) for i in range(self.num_tasks)])
        # self.experts = nn.ModuleList([Mlp(dim, dim,bias=False) for i in range(self.num_tasks)])
        print('TaskAttention',capacity_factor)


    def forward(self,x,heads=None):
        B, N, C = x.shape
        feature = x[:,self.num_tasks:,:]
        feature_multihead = feature.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #B,NH,N,C
        kv = self.kv(feature).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #2,B,NH,N,C
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)  #B,NH,N,C
        # top_k = int(feature.size(1) // self.capacity_factor)
        top_k = self.capacity_factor
        top_k = max(top_k,1)

        qs = []
        for i in range(self.num_tasks):
            tasktoken = x[:,i,:].unsqueeze(1)
            q = self.qs[i](tasktoken).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #B,NH,1,C
            qs.append(q)

        qs = torch.cat(qs,dim=2) #NT,B,NH,1,C

        atten_tf = (qs @ k.transpose(-2, -1)) * self.scale #B,NH,NT,N
        
        # atten_tf = atten_tf.softmax(dim=-1)
        atten_tf = self.attn_drop(atten_tf)

        top_k_value,top_k_indice = torch.topk(atten_tf,k=top_k,dim=-1,sorted=False) # B,NH,NT,K

        # top_k_value_softmax = top_k_value
        top_k_value_softmax = top_k_value.softmax(dim=-1)
        
        atten_topk = torch.zeros_like(atten_tf)
        atten_topk.scatter_(-1, top_k_indice,top_k_value_softmax) #B,NH,NT,N
  
        attn_token = (atten_topk @ v).transpose(1, 2).reshape(B, -1, C) #B,NT,C
        
        feature_multihead_expand = feature_multihead.unsqueeze(2).expand(-1,-1,self.num_tasks,-1,-1) #B,NH,NT,N,C
        top_k_indice_expand = top_k_indice.unsqueeze(-1).expand(-1,-1,-1,-1,v.size(-1)) #B,NH,NT,K,C
        feature_topk = torch.gather(feature_multihead_expand,-2,top_k_indice_expand) #B,NH,NT,K,C
        
        topk_value_ = top_k_value_softmax[:,:,:,:].unsqueeze(-1).expand_as(feature_topk) #B,NH,NT,K,C
        feature_topk *= topk_value_ 
        
        feature_pad = torch.zeros_like(feature_multihead_expand) #B,NH,NT,N,C
        feature_pad.scatter_(-2,top_k_indice_expand,feature_topk) #B,NH,NT,N,C
        feature_pad = feature_pad.permute(0,2,3,1,4).reshape(B,self.num_tasks,-1,C) #NT,B,N,NH*C

        feature_output = torch.stack(
            [f_i(feature_pad[ :,i, :, :]) for i, f_i in enumerate(self.experts)], dim=1 #B,NT,N,NH*C
        ) 
        token_output = torch.stack(
            [f_i(attn_token[ :,i, :]) for i, f_i in enumerate(self.experts)], dim=1  #B,NT,NH*C
        )                                         

        # for i in range(4):
        #     print(f'task {i}  top k {top_k}',(self.experts[i](attn_token[ :,i, :])).max().item(),(self.experts[i](attn_token[ :,i, :])).min().item(),(self.experts[i](feature_pad[ :,i, :, :])).max().item(),(self.experts[i](feature_pad[ :,i, :, :])).min().item())


        # task_topk_weight = torch.zeros_like(atten_tf)
        # task_topk_weight.scatter_(-1, top_k_indice,top_k_value) #B,NH,NT,N
        # # print('atten',atten_tf.max().item(),atten_tf.min().item(),self.kv.weight.max().item(),self.qs[i].weight.max().item())
        # # print(top_k,task_topk_weight.max().item(),task_topk_weight.min().item(),'ppppo',masked_softmax(task_topk_weight,dim=-2).max().item(),masked_softmax(task_topk_weight,dim=-2).min().item())
        # task_topk_weight = masked_softmax(task_topk_weight,dim=-2).permute(0,2,3,1) #B,NT,N,NH
        # task_topk_weight = task_topk_weight.repeat(1,1,1,C//self.num_heads) #B,NH,N,C

        task_topk_weight = torch.ones_like(feature_output)

        feature_output = torch.einsum('btnc,btnc->bnc',feature_output,task_topk_weight)
        output = torch.cat((token_output,feature_output),dim=1)
        # print(task_topk_weight.max().item(),atten_topk.max().item(),top_k,task_topk_weight.min().item(),atten_topk.min().item(),top_k_value.max().item(),top_k_value.min().item())
        # print('ddddd',output[:,:4].max().item(),output[:,:4].min().item())
        return output


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 num_tasks=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.num_tasks = num_tasks

#    @get_local('attn1')
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # -------------------------------------------------------------------
        mask = torch.ones_like(attn, requires_grad=False)
        if self.num_tasks > 1:
            #print(self.num_tasks,'ddddddddddddddddddddddd')
            for i in range(self.num_tasks-1):
                for j in range(i+1,self.num_tasks):
                    mask[:, :, i, j] = mask[:, :, j, i] = 0

        attn1 = attn * mask
        # attn[:,:,1,0] = attn[:,:,1,0] - 5
        # attn[:,:,1,0] = 0

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(attn1[:,:4].max(),attn1[:,:4].min(),'oopp')
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0,bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class new_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x


class ExpChoiceBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_experts=8, num_tasks=3,
                 capacity_factor=256):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = TaskAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop, proj_drop_ratio=drop,num_tasks=num_tasks,capacity_factor=capacity_factor)
        # NOTE: drop path for stochastic depth, we shall see if 
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_tasks = num_tasks
        

        self.mlp = SoftMoELayerWrapper(dim, num_experts=num_experts, slots_per_expert=1, layer=partial(Mlp, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop), normalize=True)

        self.mlp_drop = nn.Dropout(drop)
        
    
    def forward(self, x,heads=None):
        # print('ppp',x[:,:4].max().item(),x[:,:4].min().item())
        x = x + self.drop_path(self.attn(self.norm1(x),heads))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print('-------------------------------------------------')
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
    def __init__(self, patch_size=16, in_c=3, classes=[2,2,2],
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.,  norm_layer=None,
                 act_layer=None,num_tasks=1,num_experts=16,moe_gate_dim=770,capacity_factors=[256,256,256]
                 ):

        
        
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            classes (list): list of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(MultiTask_Agg, self).__init__()
        if isinstance(classes,int):
            classes = [classes] * num_tasks

        self.classes = classes
        # embed_dim = 768
        embed_dim_in = embed_dim
        self.embed_dim = embed_dim  = 768 # embed_dim for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
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
        drop_rate = drop_ratio
        attn_drop_rate = attn_drop_ratio
        self.num_tasks = num_tasks
        
        blocks = []
        
        for i in range(self.depth):
         
            blocks.append(ExpChoiceBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,num_experts=num_experts,
            drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path_ratio=drop_path_ratio, norm_layer=self.norm_layer,num_tasks=num_tasks,capacity_factor=capacity_factors[i]))
            
        print(len(blocks))
        self.blocks = nn.Sequential(*blocks)

        print(embed_dim_in,embed_dim,'sssssssssssssssssssss')
        self.projection = nn.Sequential(
            nn.Linear(embed_dim_in,embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            ) 


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.embed_dim = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.classes[1]) if classes > 0 else nn.Identity()


        #-----------------------------
        self.num_tasks = num_tasks

        self.task_tokens = nn.Parameter(torch.zeros(1, self.num_tasks, embed_dim))
        print(self.task_tokens.size(),self.task_tokens[:,0,:].reshape(1,1,embed_dim).size())


 
        self.heads = [ nn.Linear(self.embed_dim, classes[i]).cuda() for i in range(self.num_tasks) ]

        self.heads=nn.ModuleList(self.heads)

        self.pos_layer = PPEG(dim=embed_dim)
        # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.task_tokens, std=0.02)
        self.apply(_init_vit_weights)


    def forward_features(self, x):

        x = self.projection(x) # (B,N,768)

        H = x.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) #[B, N, 512]
        x = self.pos_layer(x, _H, _W)

        # print(x.size())
        task_tokens = repeat(self.task_tokens, '() n d -> b n d', b=x.size(0))        
        # print(x.size(),task_tokens.size(),'llll')
        if self.dist_token is None:
            x = torch.cat((task_tokens, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((task_tokens, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)    

        for i, blk in enumerate(self.blocks):
            x = blk(x)
                

        return x


    def forward(self, x):

        x = self.forward_features(x)
        feature = x
        # print(x.sum().item(),'www')
        res = {}
        res_lt = {}

        task_ids = list(range(self.num_tasks))

        for i in task_ids:
            x = self.norm(x)
            pred = self.heads[i](x[:,i])
            res[int(i)] = pred

        return res


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


def masked_softmax(tensor,dim=-1):
    mask = tensor != 0
    
    tensor_exp = torch.exp(tensor) * mask
    tensor_exp_sum = torch.sum(tensor_exp, dim=dim, keepdim=True) 
    tensor_exp_sum = tensor_exp_sum + (tensor_exp_sum == 0).float()
    softmax = tensor_exp / tensor_exp_sum
    
    softmax = softmax * mask
    return softmax

if __name__=='__main__':
    model = VisionTransformerMulti_LongTail_MoE(
                              patch_size=16,
                              embed_dim=512,
                              depth=4,
                              num_heads=2,
                              representation_size=None,
                              classes=2,
                              num_tasks=2,long_tail=[True,True],alpha=0.6).cuda()
    

    a=torch.FloatTensor(4,256,512).cuda()
    # model.eval()
    out=model(a)
    print(out[0].size())
    # print((out[0]==out[1]).sum()/out[1].numel())

    # print(model.heads)
    print(model.heads)