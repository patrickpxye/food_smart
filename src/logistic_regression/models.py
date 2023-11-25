from torchvision import models as models
import torch.nn as nn


def model(requires_grad, out_features, last_layer_learnable=True):
    resnet50_model = models.resnet50(progress=True, pretrained=True)
    # to freeze the hidden layers
    if not requires_grad:
        for param in resnet50_model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad:
        for param in resnet50_model.parameters():
            param.requires_grad = True
    if last_layer_learnable:
        # make the classification layer learnable
        resnet50_model.fc = nn.Linear(2048, out_features)
    return resnet50_model
