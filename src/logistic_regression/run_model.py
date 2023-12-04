import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset_builder import ImageDataset
from torch.utils.data import DataLoader
from feature_extractor.resnet50_feature_extractor import Resnet50FeatureExtractor
from sklearn.metrics import f1_score, precision_score, recall_score

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the feature extractor
extractor = Resnet50FeatureExtractor(1095).load_extractor('../../outputs/feature_model.pth')

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


def evaluate_logistic(feature_extractor, val_loader, device, num_batches=1):
    val_features = []
    val_labels = []
    batch_count = 0

    # Extract features
    with torch.no_grad():
        for data in val_loader:
            if num_batches is not None and batch_count >= num_batches:
                break  # Stop evaluation after num_batches
            inputs, labels = data['image'].to(device), data['label'].to(device)
            features = torch.sigmoid(feature_extractor(inputs)).detach().cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.detach().cpu().numpy())
            batch_count += 1

    # Concatenate all features and labels
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Convert continuous model outputs to binary labels
    predictions = (val_features > 0.5).astype(int)
        
    # Compute F1 score
    f1_micro = f1_score(val_labels, predictions, average='micro')
    precision_score_micro = precision_score(val_labels, predictions, average='micro')
    recall_score_micro = recall_score(val_labels, predictions, average='micro')

    print(f"F1 Score (Micro): {f1_micro}")
    print(f"Precision (Micro): {precision_score_micro}")
    print(f"Recall (Micro): {recall_score_micro}")

    return f1_micro, precision_score_micro, recall_score_micro


evaluate_logistic(extractor, test_loader, device, 100)

# for counter, data in enumerate(test_loader):
#     image, target = data['image'].to(device), data['label']
#     # get all the index positions where value == 1
#     target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
#     # get the predictions by passing the image through the model
#     outputs = extractor(image)
#     outputs = torch.sigmoid(outputs)
#     outputs = outputs.detach().cpu()
#     sorted_indices = np.argsort(outputs[0])
#     best = sorted_indices[-5:]
#     string_predicted = ''
#     string_actual = ''
#     for i in range(len(best)):
#         string_predicted += f"{test_data.index_to_ingredient[best[i].item()]}    "
#     for i in range(len(target_indices)):
#         string_actual += f"{test_data.index_to_ingredient[target_indices[i]]}    "
#     image = image.squeeze(0)
#     image = image.detach().cpu().numpy()
#     image = np.transpose(image, (1, 2, 0))
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
#     plt.savefig(f"../../outputs/inference_{counter}.jpg")
#     # plt.show()
