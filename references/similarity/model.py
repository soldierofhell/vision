import torch
import torch.nn as nn
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self, backbone=None):
        super(EmbeddingNet, self).__init__()
        if backbone is None:
            backbone = models.resnet101(num_classes=128, pretrained=True)

        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=1)
        return x
