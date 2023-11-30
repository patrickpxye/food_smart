from feature_extractor.feature_extractor_base import FeatureExtractorBase
from torchvision import models as models

import torch.nn as nn


class Resnet50FeatureExtractor(FeatureExtractorBase):

    def name(self):
        return "Resnet50"

    def trainable_extractor(self):
        pass
        resnet50_model = models.resnet50(progress=True, pretrained=True)
        # to freeze the hidden layers
        for param in resnet50_model.parameters():
            param.requires_grad = False

        # make the classification layer learnable
        resnet50_model.fc = nn.Linear(2048, self.out_features)
        return resnet50_model
