# UniCAS

Cervical abnormality screening is pivotal for prevention and treatment. However, the substantial size of whole slide images (WSIs) makes examination labor-intensive and time-consuming. Current deep learning-based approaches struggle with the morphological diversity of cervical cytology and require specialized models for distinct diagnostic tasks, leading to fragmented workflows. Here, we present UniCAS, a cervical cytology foundation model pre-trained via self-supervised learning on 48,532 WSIs encompassing diverse patient demographics and pathological conditions. UniCAS enables various clinical analysis tasks, achieving state-of-the-art performance in slide-level diagnosis, region-level analysis, and pixel-level image enhancement. In particular, by integrating a multi-task aggregator for slide-level diagnosis, UniCAS achieves AUCs of 92.60%, 92.58%, and 98.39% for cancer screening, candidiasis testing, and clue cell diagnosis, respectively, while reducing diagnostic time by 70% compared to conventional approaches. This work establishes a paradigm for efficient multi-scale analysis in automated cervical cytology, bridging the gap between computational pathology and clinical diagnostic workflows.

## Table of Contents

- [Requirements](#requirements)
- [Quick usage](#quick-usage)
- [Programmatic download from Hugging Face](#programmatic-download-from-hugging-face)
- [Slide-level diagnosis](#slide-level-diagnosis)
- [Citation &amp; License](#citation--license)

## Requirements

- Python 3.8+
- PyTorch
- timm (PyTorch Image Models)

Install minimal dependencies with pip:

```powershell
pip install torch torchvision timm
```

Use a CUDA-enabled PyTorch build if you want GPU acceleration.

## Quick usage

Example: create the ViT model with the UniCAS hyper-parameters and load pretrained weights you downloaded from [Hugging Face](https://huggingface.co/jianght/UniCAS).

```python
import functools
import torch
import timm

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

# build model
model = timm.models.VisionTransformer(**params)

# download the checkpoint from Hugging Face, place it locally as 'UniCAS.pth'
# then load the state dict (strict=False to allow slight architecture differences)
state = torch.load('UniCAS.pth', map_location='cpu')
model.load_state_dict(state, strict=False)

# example forward pass
model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print('Output shape:', out.shape)
```

## Slide-level diagnosis

The code for slide-level diagnosis (including the multi-task aggregator) is provided in this repository. Quick pointers:

- `models/multitask_agg.py` — implementation of the multi-task aggregator used for slide-level aggregation and multi-task prediction.
- `slide-level/train_distribute.py` — distributed training entrypoint for slide-level tasks.

More code about region-level classification, detection, and segmentation is coming soon.

## Citation & License

If you use UniCAS in your work, please credit the authors and check the model repository on Hugging Face for citation and license information.
