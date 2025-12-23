import torch
import torch.nn as nn
from models.multitask_agg import MultiTask_Agg
from torchvision import models as torchvision_models
from args import get_args
import os
from models.multitask_agg import MultiTask_Agg
from torch.nn import functional as F
from einops import rearrange
import warnings
import json
from torchvision.transforms.functional import center_crop
from einops import repeat

import numpy as np
import time



class Slide_Multitask(nn.Module):
    def __init__(self,args=None):
        super().__init__()
        self.img_size = 224
        aggregator = getattr(args,'aggregator','multi_task')
        embed_dim = getattr(args,'embed_dim',1024)
        classes = getattr(args,'classes',[2,2,2])
        num_tasks = getattr(args,'num_tasks',3)
        depth = getattr(args,'depth',8)
        gate_dim = embed_dim
        num_experts = getattr(args,'num_experts',8)
        capacity_factors =  getattr(args,'capacity_factors',256)
    
        print(f' encoder is {args.encoder}, embed_dim is {embed_dim}----------------------------------------------------------------')
        if aggregator == 'multi_task':
            self.model = MultiTask_Agg(embed_dim=embed_dim, classes=classes, num_tasks=num_tasks,depth=depth, num_experts=num_experts,moe_gate_dim=gate_dim, capacity_factors=capacity_factors)

    def forward(self, x):
        assert len(x.shape) == 3, 'Need to Feature Encoding First !'
        preds = self.model(x)
        return preds
        


def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)
