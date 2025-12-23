import os

import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DistributedSampler
import argparse

from my_dataset import EncoderData
from utils.utils import extract_feature
import pandas as pd
import random
import numpy as np
import timm
import sys
import torch
import json
from timm.layers import SwiGLUPacked
import pandas as pd 
import functools

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', default='unicas')
parser.add_argument('--csv', default='.csvs/test.csv')
parser.add_argument('--batch-size', type=int, default=2)

args = parser.parse_args()

def seed_reproducer(seed=2022):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

def main(args):
    print('-'*100)
    seed_reproducer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = EncoderData(data=pd.read_csv(args.csv), encoder=args.encoder, img_batch=50)
    print(len(dataset))
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=nw,
                                            )
    
    if args.encoder == 'UniCAS':
        params = {
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'init_values': 1e-5,
        'mlp_ratio': 2.671875 * 2,
        'mlp_layer': functools.partial(timm.layers.mlp.GluMlp, gate_last=False),
        'act_layer': torch.nn.SiLU,
        'no_embed_class': False,
        'img_size': 224,
        'num_classes': 0,
        'in_chans': 3,
        }
        model = timm.models.VisionTransformer(**params)
        state = torch.load('../UniCAS.pth')
        print(model.load_state_dict(state, strict=False),' UniCAS')

    if args.encoder == 'conch':
        from models.open_clip_custom import create_model_from_pretrained
        model,process = create_model_from_pretrained('conch_ViT-B-16',checkpoint_path='/public/home/jianght2023/checkpoints/CONCH/pytorch_model.bin')
        print(model.load_state_dict(torch.load('/public/home/jianght2023/checkpoints/CONCH/pytorch_model.bin'),strict=False),'CONCH')
        model = model.visual
    elif args.encoder == 'gigapath':
        cfg = load_cfg_from_json("/public/home/jianght2023/checkpoints/prov-gigapath/config.json")
        model = timm.create_model("vit_giant_patch14_dinov2",**cfg['model_args'],pretrained_cfg=cfg['pretrained_cfg'], pretrained=False,checkpoint_path="/public/home/jianght2023/checkpoints/prov-gigapath/pytorch_model.bin")
        print(model.load_state_dict(torch.load('/public/home/jianght2023/checkpoints/prov-gigapath/pytorch_model.bin'),strict=False),'Gigapath')
    elif args.encoder == 'hibou':
        from models.hibou import build_model
        model = build_model(weights_path="/public/home/jianght2023/checkpoints/HIBOU/hibou-b.pth")
    elif args.encoder == 'uni':
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        print(model.load_state_dict(torch.load('/public/home/jianght2023/checkpoints/UNI/pytorch_model.bin')),'UNI')
    elif args.encoder == 'hoptimus':
        params = {
            'patch_size': 14, 
            'embed_dim': 1536, 
            'depth': 40, 
            'num_heads': 24, 
            'init_values': 1e-05, 
            'mlp_ratio': 5.33334, 
            'mlp_layer': functools.partial(
                timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
            ), 
            'act_layer': torch.nn.modules.activation.SiLU, 
            'reg_tokens': 4, 
            'no_embed_class': True, 
            'img_size': 224, 
            'num_classes': 0, 
            'in_chans': 3
        }

        model = timm.models.VisionTransformer(**params)
        print(model.load_state_dict(torch.load('/public/home/jianght2023/checkpoints/Hoptimus/checkpoint.pth', map_location="cpu")),'Hoptimus')
        print(args.encoder)
    
    model = model.to(device)
    extract_feature(model=model, data_loader=loader, device=device)



def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

if __name__ == '__main__':

    main(args)
