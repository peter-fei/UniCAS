import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.nystrom_attention import NystromAttention
# from models.clip_adapter import ClipAdapter_Slide 

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512,num_tasks=1):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        self.num_tasks = num_tasks
    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, :self.num_tasks], x[:, self.num_tasks:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes,embed_dim,num_tasks=1,clip_head=False,clip_select=False):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512,num_tasks=num_tasks)
        self._fc1 = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, num_tasks, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.num_tasks = num_tasks
        # self.heads = nn.ModuleList([nn.Linear(512, self.n_classes) for i in range(num_tasks)])
        self.clip_head = clip_head
        if self.clip_head:
            self.text_projs = [ nn.Sequential(
                nn.Linear(512,512),
                nn.LayerNorm(512),
                nn.Linear(512,512)
            ) for i in range(num_tasks)]
            self.text_projs = nn.ModuleList(self.text_projs)
    
            self.heads = nn.ModuleList([nn.Linear(512, 512) for i in range(num_tasks)])
        print(f'clip_head: {clip_head} -------------------------------')
        # print('clip_select:',clip_select)
        
        self.clip_select =  ClipAdapter_Slide(embed_dim,512) if clip_select else nn.Identity()
        self.select = clip_select
        # try:
        #     self.clip_select = clip_select
        # except:
        #     self.clip_select = False
        

    def forward(self,x,text_features=None,labels=None):
        # print(x.size(),'adas')
        if self.select:
            x,loss = self.clip_select(x,labels)
                   
            print(x.size(),loss,'oopsads')

        h = x.float() #[B, n, 1024]
        # print('www',h.size())
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)

        results_dict = {}
        #---->predict
        logits = self._fc2(h[:,0]) #[B, n_classes]
        # return logits
        results_dict[0] = logits
        return results_dict

        for i in range(self.num_tasks):
            logits = self.heads[i](h[:,i]) #[B, n_classes]
            # Y_hat = torch.argmax(logits, dim=1)
            # Y_prob = F.softmax(logits, dim = 1)
            results_dict[i] = logits
            results_dict[f'feature_{i}'] = h[:,i]
            if text_features is not None:
                text_feature = self.text_projs[i](text_features[i])
                image_feature = logits / logits.norm(dim=-1, keepdim=True)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                results_dict[i] = image_feature
                results_dict[f'text_{i}'] = text_feature
                
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
