from feature_extractor.feature_extractor_base import FeatureExtractorBase
from torchvision import models
import torch.nn as nn


class VGG16FeatureExtractor(FeatureExtractorBase):

    def name(self):
        return "VGG16"

    def trainable_extractor(self):
        vgg16_model = models.vgg16(progress=True, pretrained=True)
        # to freeze the hidden layers
        for param in vgg16_model.parameters():
            param.requires_grad = False
        # make the classification layer learnable
        vgg16_model.classifier[6] = nn.Linear(4096, self.out_features)
        return vgg16_model
