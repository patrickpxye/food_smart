import models
import torch
import numpy as np

from dataset_builder import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, hamming_loss

def train_xgb(feature_extractor, train_loader, xgb_model, device, val_loader, num_batches=10):
    train_features = []
    train_labels = []
    batch_count = 0

    # Extract features
    with torch.no_grad():
        for data in train_loader:
            if num_batches is not None and batch_count >= num_batches:
                break  # Stop evaluation after num_batches
            print("Extracting features")
            inputs, labels = data['image'].to(device), data['label'].to(device)
            features = feature_extractor(inputs).detach().cpu().numpy()
            train_features.append(features)
            train_labels.append(labels.detach().cpu().numpy())
            batch_count += 1

    # Concatenate all features and labels
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Check if the dimensions of labels are correct
    if train_labels.ndim == 1 or train_labels.shape[1] != 1095:
        raise ValueError("Dimension of train_labels is not correct. Expected 1095 labels per datapoint.")

    # Train XGBoost
    for i in range(train_labels.shape[1]):  # Iterate over each class/label
        print(f"Training classifier for label {i+1}/{train_labels.shape[1]}")
        xgb_model.fit(train_features, train_labels[:, i])
    
    # Evaluation should be called after the training loop
    evaluate_xgb(feature_extractor, val_loader, xgb_model, device)

    return xgb_model


def evaluate_xgb(feature_extractor, val_loader, xgb_model, device, num_batches=1):
    val_features = []
    val_labels = []
    predictions = []
    batch_count = 0

    # Extract features
    with torch.no_grad():
        for data in val_loader:
            if num_batches is not None and batch_count >= num_batches:
                break  # Stop evaluation after num_batches
            print("Extracting features for validation")
            inputs, labels = data['image'].to(device), data['label'].to(device)
            features = feature_extractor(inputs).detach().cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.detach().cpu().numpy())
            batch_count += 1

    # Concatenate all features and labels
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Predictions for each class
    for i in range(val_labels.shape[1]):
        class_predictions = xgb_model.predict(val_features)
        predictions.append(class_predictions)
        
    # Convert predictions list to a NumPy array
    predictions = np.array(predictions).T  # Transpose to get the correct shape

    # Compute metrics
    f1_macro = f1_score(val_labels, predictions, average='macro')
    f1_micro = f1_score(val_labels, predictions, average='micro')
    hammingloss = hamming_loss(val_labels, predictions)

    print(f"F1 Score (Macro): {f1_macro}")
    print(f"F1 Score (Micro): {f1_micro}")
    print(f"Hamming Loss: {hammingloss}")

    return f1_macro, f1_micro, hammingloss


# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train dataset
# train dataset
train_data = ImageDataset("../../input/ingredients_classifier/images/",
                          "../../input/ingredients_classifier/train_images.txt",
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

# Validation data loader
val_data = ImageDataset("../../input/ingredients_classifier/images/",
                        "../../input/ingredients_classifier/val_images.txt",
                        "../../input/ingredients_classifier/val_labels.txt",
                        "../../input/ingredients_classifier/recipes.txt",
                        "../../input/ingredients_classifier/ingredients.txt",
                        False,
                        False)

val_loader = DataLoader(
    val_data,
    batch_size=32,  # Ensure this matches your validation batch size
    shuffle=False
)

# initialize the model
feature_model = models.feature_model(requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('outputs/feature_model.pth')
# Remove fully connected layer weights
checkpoint['model_state_dict'].pop('fc.weight', None)
checkpoint['model_state_dict'].pop('fc.bias', None)
# load model weights state_dict
feature_model.load_state_dict(checkpoint['model_state_dict'])
feature_model.eval()


# Integrate the XGBoost training into the training process
xgb_model = models.xgb_classifier()
xgb_model = train_xgb(feature_model, train_loader, xgb_model, device, val_loader)

xgb_model.save_model('outputs/xgb_model.json')  # Save the trained XGBoost model