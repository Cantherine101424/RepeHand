import torch
import torch.nn as nn
import timm
from main.config import cfg

class TinyViTBackbone(nn.Module):
    def __init__(self, tinyvit_type, pretrained_path):
        super(TinyViTBackbone, self).__init__()

        tinyvit_models = {
            '5m_ft': 'tiny_vit_5m_224.dist_in22k_ft_in1k',   # MACs: 1.184G, Params:  5.064M, feature shape:[1, 320, 7, 7], TOP-1: 80.7%
            '11m_ft': 'tiny_vit_11m_224.dist_in22k_ft_in1k', # MACs: 1.901G, Params: 10.536M, feature shape:[1, 448, 7, 7], TOP-1: 83.2%
            '21m_ft': 'tiny_vit_21m_224.dist_in22k_ft_in1k', # MACs: 4.079G, Params: 20.604M, feature shape:[1, 576, 7, 7], TOP-1: 84.8%
            '5m_22k': 'tiny_vit_5m_224.dist_in22k',          # MACs: 1.184G, Params:  5.064M, feature shape:[1, 320, 7, 7], TOP-1: 77.5% - 50000 images
            '11m_22k': 'tiny_vit_11m_224.dist_in22k',        # MACs: 1.901G, Params: 10.536M, feature shape:[1, 448, 7, 7], TOP-1: 80.5% - 50000 images
            '21m_22k': 'tiny_vit_21m_224.dist_in22k',        # MACs: 4.079G, Params: 20.604M, feature shape:[1, 576, 7, 7], TOP-1: 82.3% - 50000 images
            '5m_1k': 'tiny_vit_5m_224.dist_in1k',            # MACs: 1.184G, Params:  5.064M, feature shape:[1, 320, 7, 7], TOP-1: 79.1%
            '11m_1k': 'tiny_vit_11m_224.dist_in1k',          # MACs: 1.901G, Params: 10.536M, feature shape:[1, 448, 7, 7], TOP-1: 81.5%
            '21m_1k': 'tiny_vit_21m_224.dist_in1k',          # MACs: 4.079G, Params: 20.604M, feature shape:[1, 576, 7, 7], TOP-1: 83.1%
            '21m_ft_384': 'tiny_vit_21m_384.dist_in22k_ft_in1k', # MACs: 11.987G, Params: 20.604M, feature shape:[1, 576, 12, 12], TOP-1: 86.2%
            '21m_ft_512': 'tiny_vit_21m_512.dist_in22k_ft_in1k', # MACs: 21.310G, Params: 20.604M, feature shape:[1, 576, 16, 16], TOP-1: 86.5%

        }

        model_name = tinyvit_models[tinyvit_type]

        self.model = timm.create_model(model_name, pretrained=True)
        self.pretrained_path=pretrained_path

        self.model.head = nn.Identity()

    def forward(self, x):
        x = self.model.forward_features(x)
        return x

    def init_weights(self):
        model_zoo_url = self.pretrained_path
        org_tinyvit = torch.load(model_zoo_url, map_location=torch.device('cpu'))

        org_tinyvit.pop('head.norm.weight', None)
        org_tinyvit.pop('head.norm.bias', None)
        org_tinyvit.pop('head.fc.weight', None)
        org_tinyvit.pop('head.fc.bias', None)

        new_state_dict = {k.replace('module.', ''): v for k, v in org_tinyvit.items()}

        self.load_state_dict(new_state_dict, strict=False)
        print("Initialize tiny_vit from model zoo")

if __name__ == "__main__":
    # backbone = TinyViTBackbone('21m_ft', cfg.pretrained_path)
    backbone = TinyViTBackbone('21m_ft', cfg.pretrained_path)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = backbone(input_tensor)
    print(output.shape)

    from thop import profile, clever_format
    macs, params = profile(backbone, inputs=(input_tensor, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")