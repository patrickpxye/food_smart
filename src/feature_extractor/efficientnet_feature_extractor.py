
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from feature_extractor.feature_extractor_base import FeatureExtractorBase

class EfficientNetFeatureExtractor(FeatureExtractorBase):

    def name(self):
        return "EfficientNet"

    def trainable_extractor(self):
        efficientnet_model = EfficientNet.from_pretrained('efficientnet-b7')
        # to freeze the hidden layers
        for param in efficientnet_model.parameters():
            param.requires_grad = False
        # make the classification layer learnable
        efficientnet_model._fc = nn.Linear(efficientnet_model._fc.in_features, self.out_features)

        return efficientnet_model
