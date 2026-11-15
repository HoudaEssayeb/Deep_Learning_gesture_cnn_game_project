
import torch.nn as nn
from torchvision import models


CLASS_NAMES = [
    "idle",
    "jump",
    "left",
    "right",
    "roll"
]


def build_model(num_classes=len(CLASS_NAMES), freeze_backbone=True):
    """
    Build a MobileNetV2 model using transfer learning.
    """

    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze pretrained feature extraction layers
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Adapt classifier to our gesture classes
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model
