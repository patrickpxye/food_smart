import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_builder import ImageDataset
from torch.utils.data import DataLoader

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
feature_model = models.feature_model(requires_grad=False).to(device)
checkpoint_feature = torch.load('outputs/feature_model.pth')
checkpoint_feature['model_state_dict'].pop('fc.weight', None)
checkpoint_feature['model_state_dict'].pop('fc.bias', None)
# load model weights state_dict
feature_model.load_state_dict(checkpoint_feature['model_state_dict'])
feature_model.eval()

xgb_model = models.xgb_classifier()
xgb_model.load_model('outputs/xgb_model.json')

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

def predict_labels(xgb_model, features, threshold=0.5):
    # Predict label probabilities
    label_probabilities = xgb_model.predict_proba(features)
    # Apply threshold to get predictions
    label_predictions = (label_probabilities >= threshold).astype(int)
    return label_predictions

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    
    # Extract features using the feature model
    features = feature_model(image).detach().cpu().numpy()
    
    # Predict labels using the XGBoost model
    label_predictions = predict_labels(xgb_model, features)
    
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
