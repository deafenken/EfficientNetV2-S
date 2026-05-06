import torch
from torch import nn
from torchvision import models


def create_model(num_classes, name="resnet18", pretrained=False):
    if name != "resnet18":
        raise ValueError("Only resnet18 is implemented in the first baseline")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

