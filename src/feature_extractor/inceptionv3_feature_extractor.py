import torch.nn as nn
from torchvision import models
from feature_extractor.feature_extractor_base import FeatureExtractorBase


class InceptionV3FeatureExtractor(FeatureExtractorBase):

    def name(self):
        return "InceptionV3"

    def trainable_extractor(self):
        inception_model = models.inception_v3(progress=True, pretrained=True)
        # to freeze the hidden layers
        for param in inception_model.parameters():
            param.requires_grad = False
        # make the classification layer learnable
        inception_model.fc = nn.Linear(2048, self.out_features)
        return inception_model
