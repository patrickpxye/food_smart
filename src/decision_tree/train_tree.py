import models
import torch
import numpy as np
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt

from dataset.dataset_builder import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, hamming_loss
from feature_extractor.resnet50_feature_extractor import Resnet50FeatureExtractor
from feature_extractor.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from feature_extractor.vgg16_feature_extractor import VGG16FeatureExtractor
from feature_extractor.inceptionv3_feature_extractor import InceptionV3FeatureExtractor

def train_xgb(feature_extractor, train_loader, device, val_loader, num_batches=100):
    train_features = []
    train_labels = []
    batch_count = 0
    evals_results = []

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

    val_features = []
    val_labels = []
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

    # Train XGBoost for each label
    xgb_models = []  # List to store models
    for i in range(train_labels.shape[1]):  # Iterate over each class/label
        print(f"Training classifier for label {i+1}/{train_labels.shape[1]}")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=100,
            learning_rate=0.1,
            use_label_encoder=False  # to avoid a warning since XGBoost 1.3.0 release
        )  # Create a new instance for each label
        xgb_model.fit(train_features, train_labels[:, i],
                      eval_set=[(train_features, train_labels[:, i]), (val_features, val_labels[:, i])],
                      eval_metric=['logloss', 'error'],
                      early_stopping_rounds=10,
                      verbose=True)
        evals_results.append(xgb_model.evals_result())  # Store evaluation results
        xgb_models.append(xgb_model)  # Store the trained model

    return xgb_models, evals_results

def evaluate_xgb(feature_extractor, val_loader, xgb_models, device, num_batches=1):
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
    predictions = []
    for model in xgb_models:  # Use each model for prediction
        class_predictions = model.predict(val_features)
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
                          "../../input/ingredients_classifier/filtered_ingredients.txt",
                          True,
                          False)
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
                        "../../input/ingredients_classifier/filtered_ingredients.txt",
                        False,
                        False)
val_loader = DataLoader(
    val_data,
    batch_size=32,  # Ensure this matches your validation batch size
    shuffle=False
)

# # initialize the model
# feature_model = models.feature_model(requires_grad=False).to(device)
# # load the model checkpoint
# checkpoint = torch.load('trained_models/feature_model.pth')
# # Remove fully connected layer weights
# checkpoint['model_state_dict'].pop('fc.weight', None)
# checkpoint['model_state_dict'].pop('fc.bias', None)
# # load model weights state_dict
# feature_model.load_state_dict(checkpoint['model_state_dict'])
# feature_model.eval()

#feature_model = InceptionV3FeatureExtractor(1095).load_extractor('../feature_extractor/InceptronV3_feature_extractor.pth')
#feature_model = EfficientNetFeatureExtractor(1095).load_extractor('../../outputs/efficientNet_feature_extractor.pth')
feature_model = Resnet50FeatureExtractor(255).load_extractor('../../outputs/resnet50_feature_extractor_255.pth')


def plot_metrics(evals_results):
    for idx, evals_result in enumerate(evals_results):
        epochs = len(evals_result['validation_0']['error'])
        x_axis = range(0, epochs)

        # Plot log loss
        plt.figure()
        plt.plot(x_axis, evals_result['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, evals_result['validation_1']['logloss'], label='Validation')
        plt.legend()
        plt.ylabel('Log Loss')
        plt.title(f'XGBoost Log Loss for Classifier {idx+1}')
        plt.show()

        # Plot classification error
        plt.figure()
        plt.plot(x_axis, evals_result['validation_0']['error'], label='Train')
        plt.plot(x_axis, evals_result['validation_1']['error'], label='Validation')
        plt.legend()
        plt.ylabel('Classification Error')
        plt.title(f'XGBoost Classification Error for Classifier {idx+1}')
        plt.show()

# def plot_tree_structure(model, num_trees=0):
#     plot_tree(model, num_trees=num_trees)
#     plt.show()


# Integrate the XGBoost training into the training process
xgb_models, evals_results = train_xgb(feature_model, train_loader, device, val_loader)
# plot_metrics(evals_result)
# plot_tree_structure(xgb_models[0], num_trees=0)

def plot_combined_aggregated_loss(evals_results):
    # Aggregate losses
    max_epochs = max(len(e['validation_0']['logloss']) for e in evals_results)
    aggregated_loss = {'train': [0]*max_epochs, 'validation': [0]*max_epochs}

    for result in evals_results:
        for epoch in range(max_epochs):
            if epoch < len(result['validation_0']['logloss']):
                aggregated_loss['train'][epoch] += result['validation_0']['logloss'][epoch]
                aggregated_loss['validation'][epoch] += result['validation_1']['logloss'][epoch]

    # Average the loss
    num_models = len(evals_results)
    aggregated_loss['train'] = [loss / num_models for loss in aggregated_loss['train']]
    aggregated_loss['validation'] = [loss / num_models for loss in aggregated_loss['validation']]

    # Plotting
    epochs = len(aggregated_loss['train'])
    x_axis = range(0, epochs)

    plt.figure()
    plt.plot(x_axis, aggregated_loss['train'], label='Average Train Loss')
    plt.plot(x_axis, aggregated_loss['validation'], label='Average Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('Average Log Loss Across All XGBoost Models')
    plt.savefig('outputs/average_loss_res.pdf')
    plt.show()
    

# Usage
plot_combined_aggregated_loss(evals_results)


# Save each trained XGBoost model
for i, model in enumerate(xgb_models):
    model.save_model(f'trained_models_resnet50/xgb_model_{i}.json')