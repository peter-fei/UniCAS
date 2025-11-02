How to use UniCAS:

params = {
    'patch_size': 16, 
    'embed_dim': 1024, 
    'depth': 24, 
    'num_heads': 16, 
    'init_values': 1e-05, 
    'mlp_ratio': 2.671875 * 2, 
    'mlp_layer': functools.partial(
        timm.layers.mlp.GluMlp, gate_last=False
    ), 
    'act_layer': torch.nn.modules.activation.SiLU, 
    'no_embed_class': False, 
    'img_size': 224, 
    'num_classes': 0, 
    'in_chans': 3
}

model = timm.models.VisionTransformer(**params)
model_dict = torch.load('UniCAS.pth')
print(model2.load_state_dict(model_dict,strict=False),'UniCAS')
