import sys
import os

# Add the parent directory to the sys.path list
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# from engine import train, validate
from dataset.dataset_builder import ImageDataset
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from collections import defaultdict
from joblib import load
import numpy as np
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score


matplotlib.style.use('ggplot')


# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
# train dataset
train_data = ImageDataset(
                        "../../input/ingredients_classifier/images/",
                        "../../input/ingredients_classifier/train_images.txt",
                       "../../input/ingredients_classifier/train_labels.txt",
                       "../../input/ingredients_classifier/recipes.txt",
                       "../../input/ingredients_classifier/ingredients.txt",
                       True,
                       False
                       )
# validation dataset
valid_data = ImageDataset(
                        "../../input/ingredients_classifier/images/",
                        "../../input/ingredients_classifier/val_images.txt",
                       "../../input/ingredients_classifier/val_labels.txt",
                       "../../input/ingredients_classifier/recipes.txt",
                       "../../input/ingredients_classifier/ingredients.txt",
                       False,
                       False, 
                       num_samples=32)
def most_common_label(labels):
    return max(set(labels), key=labels.count)
# train data loader
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False
)

## Load the pretrained ResNet50 model
pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
pretrained_resnet = torch.nn.Sequential(*(list(pretrained_resnet.children())[:-1]))
pretrained_resnet.eval()

# Load the saved K-Means model
kmeans = load('kmeans_model.joblib')

# Load the saved cluster labels
saved_cluster_labels = np.load('cluster_labels.npy')

# Later, to load the dictionary
with open('most_common_labels.pkl', 'rb') as fp:
    most_common_labels = pickle.load(fp)
    
# Extract the label matrix (ingredients matrix)
ingredient_matrix = np.array([train_data[i]['label'] for i in range(len(train_data))])

# Load the saved K-Means model
kmeans_ingredients = load('kmeans_model_ingredients.joblib')

# Load the saved cluster labels
saved_cluster_labels = np.load('cluster_labels_ingredients.npy')


with open('most_common_labels_ingredients.pkl', 'rb') as fp:
    ingredient_cluster_labels = pickle.load(fp)
    # print("ingredient cluster labels", ingredient_cluster_labels)
    
def extract_features(model, dataloader):
    features = []
    total_batches = len(dataloader)  # Total number of batches in the dataloader

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data['image']
            output = model(inputs)
            features.extend(output.squeeze().cpu().numpy())
            # Print the progress
            print(f"Finished extracting features from batch {i+1}/{total_batches}")
    return features

features = extract_features(pretrained_resnet, valid_loader)

predicted_clusters = kmeans.predict(features)
# print("predicted clusters", predicted_clusters)

def compute_metrics(actual_labels, predicted_labels):
    f1_scores = []
    precisions = []
    recalls = []
    
    for actual, predicted in zip(actual_labels, predicted_labels):
        union_labels = actual.union(predicted)
        actual_binary = [1 if label in actual else 0 for label in union_labels]
        predicted_binary = [1 if label in predicted else 0 for label in union_labels]

        f1 = f1_score(actual_binary, predicted_binary, average='micro')
        precision = precision_score(actual_binary, predicted_binary, average='micro')
        recall = recall_score(actual_binary, predicted_binary, average='micro')

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        # Debugging print
        print(f"Actual: {actual}, Predicted: {predicted}, F1: {f1}, Precision: {precision}, Recall: {recall}")

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    return avg_f1, avg_precision, avg_recall

actual_labels = []
predicted_labels = []


for counter, data in enumerate(valid_loader):
    images, targets = data['image'].to(device), data['label']
    
    # Loop through each image in the batch
    for i in range(images.size(0)):
        image = images[i].squeeze().detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))

        predicted_label_name = most_common_labels[predicted_clusters[counter * images.size(0) + i]]
        print("predicted label name", predicted_label_name)
        
        # indices for the clusters of  kmeans_ingredients     
        unique_cluster_indices = [index for index, item in  ingredient_cluster_labels.items() if item == predicted_label_name]
        print("unique cluster indices", unique_cluster_indices)
         # Aggregate all ingredients from these clusters
         
        all_ingredients = set()

        for index in unique_cluster_indices:
            # Get the labels assigned by the KMeans model
            labels = kmeans_ingredients.labels_

            # Get the indices of items that belong to cluster 'index'
            indices_in_cluster = np.where(labels == index)[0]
            print("indices in cluster", indices_in_cluster)

            # Retrieve the ingredient names belonging to this cluster
            for idx in indices_in_cluster:
                if idx in train_data.index_to_ingredient:
                    ingredient_name = train_data.index_to_ingredient[idx]
                    all_ingredients.add(ingredient_name)
                else:
                    print(f"Index {idx} not found in index_to_ingredient.")

        # Convert the set of all ingredients to a list and then to a string
        predicted_ingredients = '    '.join(list(all_ingredients))
        print("Predicted Ingredients:", predicted_ingredients)
        target_indices = [j for j in range(len(targets[i])) if targets[i][j] == 1]
        actual_label_names = [valid_data.index_to_ingredient[idx] for idx in target_indices]
        print("Actual Ingredients:", actual_label_names)

        # Plotting the image
        plt.imshow(image)
        plt.axis('off')
        string_predicted = f"{predicted_ingredients} "
        string_actual = '    '.join(actual_label_names)
        plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
        plt.savefig(f"outputs_k_means/inference_{counter}_{i}.jpg")
        plt.close()

        actual_labels.append(set(actual_label_names))
        predicted_labels.append(all_ingredients)
        
# After the loop
average_f1, average_precision, average_recall = compute_metrics(actual_labels, predicted_labels)
print(f"Average F1 Score: {average_f1}")
print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")