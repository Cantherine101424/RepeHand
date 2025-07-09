import torch
import torch.nn as nn
import timm

class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type):
        super(ResNetBackbone, self).__init__()

        resnet_models = {
            18: 'resnet18',
            34: 'resnet34',
            50: 'resnet50',
            101: 'resnet101',
            152: 'resnet152'
        }

        model_name = resnet_models[resnet_type]

        self.model = timm.create_model(model_name, pretrained=True)

        self.model.fc = nn.Identity()

    def forward(self, x):

        x = self.model.forward_features(x)

        return x

    def init_weights(self):
        self.model = timm.create_model(self.model.default_cfg['architecture'], pretrained=True)
        print("Initialize resnet from model zoo")

if __name__ == "__main__":
    backbone = ResNetBackbone(50)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = backbone(input_tensor)
    print(output.shape)

    from thop import profile, clever_format
    macs, params = profile(backbone, inputs=(input_tensor, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    print(f"  MACs: {macs}")
    print(f"Params: {params}")