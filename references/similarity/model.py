import torch
import torch.nn as nn
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self, backbone=None):
        super(EmbeddingNet, self).__init__()
        if backbone is None:
            model = ResNet(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=128)
            state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
            for key in list(state_dict.keys):
                if key in ["fc.weight", "fc.bias"]:
                    state_dict.pop(key)
            
            model.load_state_dict(state_dict, strict=False)
            
            #backbone = models.resnet101(num_classes=128)

        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=1)
        return x
