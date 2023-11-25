import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset.dataset_builder import ImageDataset
from torch.utils.data import DataLoader
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

def predict_labels(xgb_models, features, threshold=0.3):
    # Aggregate predictions from all models
    label_predictions = np.zeros((features.shape[0], len(xgb_models)))
    for i, model in enumerate(xgb_models):
        # Predict label probabilities for each model
        label_probabilities = model.predict_proba(features)[:, 1]  # Assuming binary classification
        label_predictions[:, i] = label_probabilities
    
    # Apply threshold to get final predictions
    label_predictions = (label_predictions >= threshold).astype(int)
    return label_predictions


for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    
    # Extract features using the feature model
    features = feature_model(image).detach().cpu().numpy()
    
    # Predict labels using the XGBoost model
    label_predictions = predict_labels(xgb_models, features)
    
    # Get indices of the predicted labels
    predicted_indices = np.where(label_predictions[0] == 1)[0]
    
    # Convert indices to corresponding ingredient names
    string_predicted = '    '.join([test_data.index_to_ingredient[idx] for idx in predicted_indices])
    string_actual = '    '.join([test_data.index_to_ingredient[idx] for idx in target_indices])
    
    # Process and plot the image
    image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(f"outputs/inference_{counter}.jpg")
    plt.show()
