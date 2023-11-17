from torchvision import models as models
import torch.nn as nn
import xgboost as xgb

def feature_model(pretrained, requires_grad):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    model.fc = nn.Identity()
    return model

def xgb_classifier(n_estimators=100, max_depth=6, learning_rate=0.1):
    # Each label is treated as a separate binary classification problem.
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False  # to avoid a warning since XGBoost 1.3.0 release
    )
    # Placeholder for fitting the classifier; to be used during the training loop
    return xgb_classifier
