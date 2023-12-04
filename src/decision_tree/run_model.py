import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset.dataset_builder import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
feature_model = models.feature_model(requires_grad=False).to(device)
checkpoint_feature = torch.load('trained_models/feature_model.pth')
checkpoint_feature['model_state_dict'].pop('fc.weight', None)
checkpoint_feature['model_state_dict'].pop('fc.bias', None)
# load model weights state_dict
feature_model.load_state_dict(checkpoint_feature['model_state_dict'])
feature_model.eval()

num_models = 1095  # Assuming you have 1095 models, one for each class
xgb_models = []
for i in range(num_models):
    model= xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=100,
            learning_rate=0.1,
            use_label_encoder=False  # to avoid a warning since XGBoost 1.3.0 release
        )
    model.load_model(f'trained_models/xgb_model_{i}.json')
    xgb_models.append(model)

# prepare the test dataset and dataloader
test_data = ImageDataset("../../input/ingredients_classifier/images/",
                         "../../input/ingredients_classifier/test_images.txt",
                         "../../input/ingredients_classifier/test_labels.txt",
                         "../../input/ingredients_classifier/recipes.txt",
                         "../../input/ingredients_classifier/ingredients.txt",
                         False,
                         True)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False
)

def evaluate_xgb(feature_extractor, val_loader, xgb_models, device, num_batches=1, threshold=0.1):
    val_features = []
    val_labels = []
    predictions = []
    batch_count = 0

    # Extract features
    with torch.no_grad():
        for data in val_loader:
            if num_batches is not None and batch_count >= num_batches:
                break  # Stop evaluation after num_batches
            print("Extracting features for evaluation")
            inputs, labels = data['image'].to(device), data['label'].to(device)
            features = feature_extractor(inputs).detach().cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.detach().cpu().numpy())
            batch_count += 1

    # Concatenate all features and labels
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Predictions for each class
    predictions = []
    for model in xgb_models:  # Use each model for prediction
        prob_predictions = model.predict_proba(val_features)
        class_predictions = (prob_predictions[:, 1] >= threshold).astype(int)
        predictions.append(class_predictions)
        # class_predictions = model.predict(val_features)
        # predictions.append(class_predictions)
        
    # Convert predictions list to a NumPy array
    predictions = np.array(predictions).T  # Transpose to get the correct shape

    # Compute metrics
    f1_micro = f1_score(val_labels, predictions, average='micro')
    precision_micro = precision_score(val_labels, predictions, average='micro')
    recall_micro = recall_score(val_labels, predictions, average='micro')

    print(f"F1 Score (Micro): {f1_micro}")
    print(f"Precision (Micro): {precision_micro}")
    print(f"Recall (Micro): {recall_micro}")

    return f1_micro


evaluate_xgb(feature_model, test_loader, xgb_models, device, 100 , 0.2)


def predict_labels(xgb_models, features, threshold=0.1):
    # Aggregate predictions from all models
    label_predictions = np.zeros((features.shape[0], len(xgb_models)))
    for i, model in enumerate(xgb_models):
        # Predict label probabilities for each model
        label_probabilities = model.predict_proba(features)[:, 1]  # Assuming binary classification
        label_predictions[:, i] = label_probabilities
    
    # Apply threshold to get final predictions
    label_predictions = (label_predictions >= threshold).astype(int)
    return label_predictions

# for counter, data in enumerate(test_loader):
#     image, target = data['image'].to(device), data['label']
#     target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    
#     # Extract features using the feature model
#     features = feature_model(image).detach().cpu().numpy()
    
#     # Predict labels using the XGBoost model
#     label_predictions = predict_labels(xgb_models, features)
    
#     # Get indices of the predicted labels
#     predicted_indices = np.where(label_predictions[0] == 1)[0]
    
#     # Convert indices to corresponding ingredient names
#     string_predicted = '    '.join([test_data.index_to_ingredient[idx] for idx in predicted_indices])
#     string_actual = '    '.join([test_data.index_to_ingredient[idx] for idx in target_indices])
    
#     # Process and plot the image
#     image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
#     plt.savefig(f"outputs/inference_{counter}.jpg")
#     plt.show()
