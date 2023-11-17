import models
import torch
import numpy as np

from dataset_builder import ImageDataset
from torch.utils.data import DataLoader

# Train XGBoost on extracted features
def train_xgb(feature_extractor, train_loader, xgb_model, device):
    
    train_features = []
    train_labels = []

    # Extract features
    with torch.no_grad():
        for data in train_loader:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            features = feature_extractor(inputs).detach().cpu().numpy()
            train_features.append(features)
            train_labels.append(labels.detach().cpu().numpy())

    # Concatenate all features and labels
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Train XGBoost
    for i in range(train_labels.shape[1]):  # Iterate over each class/label
        # Train a separate classifier for each label
        xgb_model.fit(train_features, train_labels[:, i])

    return xgb_model

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train dataset
train_data = ImageDataset("../../input/ingredients_classifier/train_images.txt",
                       "../../input/ingredients_classifier/train_labels.txt",
                       "../../input/ingredients_classifier/recipes.txt",
                       "../../input/ingredients_classifier/ingredients.txt",
                       True,
                       False)

# train data loader
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

# Integrate the XGBoost training into the training process
xgb_model = models.xgb_classifier()
xgb_model = train_xgb(models.feature_model, train_loader, xgb_model, device)

torch.save(xgb_model, '../outputs/xgb_model.pth')  # Save the trained XGBoost model