import models
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataset.dataset_builder import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import xgboost as xgb
from feature_extractor.resnet50_feature_extractor import Resnet50FeatureExtractor
from feature_extractor.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from feature_extractor.vgg16_feature_extractor import VGG16FeatureExtractor
from feature_extractor.inceptionv3_feature_extractor import InceptionV3FeatureExtractor

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #intialize the model
# feature_model = models.feature_model(requires_grad=False).to(device)
# checkpoint_feature = torch.load('../../outputs/vgg16_feature_extractor.pth')
# checkpoint_feature['model_state_dict'].pop('fc.weight', None)
# checkpoint_feature['model_state_dict'].pop('fc.bias', None)
# # load model weights state_dict
# feature_model.load_state_dict(checkpoint_feature['model_state_dict'])
# feature_model.eval()

feature_model = Resnet50FeatureExtractor(255).load_extractor('../../outputs/resnet50_feature_extractor_255.pth')
#feature_model = EfficientNetFeatureExtractor(1095).load_extractor('../../outputs/efficientNet_feature_extractor.pth')
#feature_model = InceptionV3FeatureExtractor(1095).load_extractor('../feature_extractor/InceptronV3_feature_extractor.pth')
#feature_model = VGG16FeatureExtractor(1095).load_extractor('../../outputs/vgg16_feature_extractor.pth')

num_models = 255  # Assuming you have 1095 models, one for each class
xgb_models = []
for i in range(num_models):
    model= xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=100,
            learning_rate=0.1,
            use_label_encoder=False  # to avoid a warning since XGBoost 1.3.0 release
        )
    model.load_model(f'trained_models_resnet50/xgb_model_{i}.json')
    xgb_models.append(model)

# prepare the test dataset and dataloader
test_data = ImageDataset("../../input/ingredients_classifier/images/",
                         "../../input/ingredients_classifier/test_images.txt",
                         "../../input/ingredients_classifier/test_labels.txt",
                         "../../input/ingredients_classifier/recipes.txt",
                         "../../input/ingredients_classifier/filtered_ingredients.txt",
                         False,
                         True)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False
)

def evaluate_xgb(feature_extractor, val_loader, xgb_models, device, num_batches=1, threshold=0.5):
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
        
    # Convert predictions list to a NumPy array
    predictions = np.array(predictions).T  # Transpose to get the correct shape

    # Compute metrics
    f1_micro = f1_score(val_labels, predictions, average='micro')
    precision_micro = precision_score(val_labels, predictions, average='micro')
    recall_micro = recall_score(val_labels, predictions, average='micro')
    accuracy = accuracy_score(val_labels.T, predictions.T)


    # Initialize lists to store the average metrics
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    precisions = []
    recalls = []
    average_precisions = []
    mean_recall = np.linspace(0, 1, 100)
    cumulative_conf_matrix = np.zeros((2, 2))

    # Compute ROC and PR for each class
    for i, model in enumerate(xgb_models):
        y_true = val_labels[:, i]
        y_scores = predictions[:, i]

        # Update the cumulative confusion matrix
        conf_matrix = confusion_matrix(y_true, y_scores)
        cumulative_conf_matrix += conf_matrix

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        rev_recall = np.fliplr([recall])[0]  # Reverse the array
        rev_precision = np.fliplr([precision])[0]
        precisions.append(np.interp(mean_recall, rev_recall, rev_precision))
        average_precisions.append(average_precision_score(y_true, y_scores))

    # Average ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # Average Precision-Recall
    mean_precision = np.mean(precisions, axis=0)

    # Plot Average ROC
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Average ROC (AUC = {mean_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve across all classes')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Average Precision-Recall
    plt.figure()
    plt.plot(mean_recall, mean_precision, color='b', label=f'Average Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall Curve across all classes')
    plt.legend(loc="lower left")
    plt.show()

    # Plot the cumulative confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cumulative_conf_matrix/255, annot=True, fmt=".2f", cmap="Blues")  # Using .2f for formatting
    plt.title("Cumulative Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    #plt.savefig('outputs/confusion_matrix.png')

    print(f"F1 Score (Micro): {f1_micro}")
    print(f"Precision (Micro): {precision_micro}")
    print(f"Recall (Micro): {recall_micro}")
    print(f"Accuracy: {accuracy}")


    return f1_micro, precision_micro, recall_micro


evaluate_xgb(feature_model, test_loader, xgb_models, device, 100, 0.25)


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
