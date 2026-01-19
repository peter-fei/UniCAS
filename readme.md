# UniCAS: A Foundation Model for Cervical Cytology Screening

UniCAS is a foundation model pre-trained on **48,532 cervical whole slide images (WSIs)**, designed to handle the morphological diversity of cervical cytology. It achieves state-of-the-art performance in:

- **Slide-level diagnosis** (Cancer screening, Candidiasis, Clue cells)
- **Region-level analysis** (Detection & Segmentation)
- **Pixel-level image enhancement**

By integrating a multi-task aggregator, UniCAS achieves AUCs of **92.60% (Cancer)**, **92.58% (Candidiasis)**, and **98.39% (Clue cells)**, reducing diagnostic time by 70%.

## Requirements

- Python 3.9+
- PyTorch
- `timm==1.0.5`

## Quick Usage

### 1. Download Weights

Download the pre-trained weights (`UniCAS.pth`) from:

- [Baidu Netdisk](https://pan.baidu.com/s/154QIsDYiBjDaxhyvlc2YYQ?pwd=9z95) (Password: `9z95`)
- [Hugging Face](https://huggingface.co/jianght/UniCAS)

### 2. Load Model

```python
import functools
import torch
import timm

# Define UniCAS parameters
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

# Build model
model = timm.models.VisionTransformer(**params)

# Load checkpoint
# Ensure 'UniCAS.pth' is in your current directory
state = torch.load('UniCAS.pth', map_location='cpu')
print(model.load_state_dict(state, strict=False))

# Inference example
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

with torch.no_grad():
    # Input: [Batch, Channels, Height, Width]
    x = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
    out = model(x)
    print('Output feature shape:', out.shape) # Expected: [1, 1024]
```

## Task-Specific Guides

### Slide-level Diagnosis

Located in the `slide-level/` directory.

- **Aggregator**: [`slide-level/models/multitask_agg.py`](slide-level/models/multitask_agg.py)
- **Training**: [`slide-level/train_distribute.py`](slide-level/train_distribute.py)
- **Detailed Guide**: See [slide-level/README.md](slide-level/README.md) for data preparation and training instructions.

### Region-level Analysis

For detection and segmentation tasks, we utilize [Detectron2](https://github.com/facebookresearch/detectron2).
*(Note: Refer to specific sub-directories or future updates for integration code.)*

## Citation & License

If you use UniCAS in your research, please credit the authors.