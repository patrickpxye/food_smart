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
from collections import defaultdict, Counter
from joblib import dump
import pickle

matplotlib.style.use('ggplot')

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# intialize the model
# learning parameters
# lr = 0.0001
# epochs = 20
batch_size = 32
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.BCELoss()

# train dataset
train_data = ImageDataset(
                    "../../input/ingredients_classifier/images/",
                    "../../input/ingredients_classifier/train_images.txt",
                       "../../input/ingredients_classifier/train_labels.txt",
                       "../../input/ingredients_classifier/recipes.txt",
                       "../../input/ingredients_classifier/filtered_ingredients.txt",
                       True,
                       False
                       )
# validation dataset
valid_data = ImageDataset(
                        "../../input/ingredients_classifier/images/",
                        "../../input/ingredients_classifier/val_images.txt",
                       "../../input/ingredients_classifier/val_labels.txt",
                       "../../input/ingredients_classifier/recipes.txt",
                       "../../input/ingredients_classifier/filtered_ingredients.txt",
                       False,
                       False)
print("finished loading data")

def most_common_label(labels):
    return max(set(), key=labels.count)

def top_n_common_labels(labels, n=5):
    # Count the occurrences of each label
    label_counts = Counter(labels)
    
    # Get the top n most common labels
    # The most_common method returns a list of tuples (label, count)
    top_n = label_counts.most_common(n)

    # Extract just the labels from the tuples
    top_n_labels = [label for label, count in top_n]

    return top_n_labels

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


######################################### cluster images #########################################
## Load the pretrained ResNet50 model
pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
pretrained_resnet = torch.nn.Sequential(*(list(pretrained_resnet.children())[:-1]))
pretrained_resnet.eval()

# Function to extract features using the pretrained model
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

# Your DataLoader for the images
image_dataloader = train_loader 
print("extracting features")
# Extract features
features = extract_features(pretrained_resnet, image_dataloader)

print("finished getting features")
# Perform clustering on the extracted features
num_clusters = 100 
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(features)
print("finished clustering")

# Save the entire K-Means model
dump(kmeans, 'saved_outputs/kmeans_model.joblib')
# Save the cluster labels
np.save('saved_outputs/cluster_labels.npy', cluster_labels)

# Convert the list of features to a NumPy array
features_array = np.array(features)

# Ensure the perplexity is appropriate for the number of samples
perplexity_value = min(30, len(features_array) - 1)  # For example, 30 or less

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
reduced_features = tsne.fit_transform(features_array)


data_labels = [train_data.get_label(i) for i in range(train_data.__len__())]

# Collect labels for each cluster
cluster_to_labels = defaultdict(list)
for label, cluster_id in zip(data_labels, cluster_labels):
    cluster_to_labels[cluster_id].append(label)

# Find the most common label for each cluster
most_common_labels = {cluster: most_common_label(labels) for cluster, labels in cluster_to_labels.items()}

# Save the dictionary using Pickle
with open('saved_outputs/most_common_labels.pkl', 'wb') as fp:
    pickle.dump(most_common_labels, fp)
# Calculate cluster centers
cluster_centers = {cluster: np.mean(reduced_features[cluster_labels == cluster], axis=0) for cluster in most_common_labels}

# Plotting
plt.figure(figsize=(12, 8))
for cluster_id, center in cluster_centers.items():
    # Plotting each cluster with its points
    points = reduced_features[cluster_labels == cluster_id]
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster_id} ({most_common_labels[cluster_id]})')

    # Annotating the cluster center with the most common label
    label = most_common_labels[cluster_id]
    plt.annotate(label, (center[0], center[1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('K-Means Clustering with t-SNE visualization')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
# plt.legend()
plt.show()

######################################### end  #########################################


######################################### Cluster labels   #########################################
# Extract the label matrix (ingredients matrix)
ingredient_matrix = np.array([train_data[i]['label'] for i in range(len(train_data))])

# Perform clustering on the ingredient matrix
num_clusters = 100  # Adjust based on your needs
kmeans_ingredients = KMeans(n_clusters=num_clusters, random_state=0)
ingredient_cluster_labels = kmeans_ingredients.fit_predict(ingredient_matrix)

# Save the entire K-Means model
dump(kmeans_ingredients, 'saved_outputs/kmeans_model_ingredients.joblib')
# Save the cluster labels
np.save('saved_outputs/cluster_labels_ingredients.npy', ingredient_cluster_labels)

data_labels = [train_data.get_label(i) for i in range(train_data.__len__())]
# Collect labels for each cluster
cluster_to_labels = defaultdict(list)
for label, cluster_id in zip(data_labels, ingredient_cluster_labels):
    cluster_to_labels[cluster_id].append(label)

# Find the most common label for each cluster
def most_common_label(labels):
    return max(set(labels), key=labels.count)

most_common_labels_ingredients = {cluster: most_common_label(labels) for cluster, labels in cluster_to_labels.items()}

# Save the dictionary using Pickle
with open('saved_outputs/most_common_labels_ingredients.pkl', 'wb') as fp:
    pickle.dump(most_common_labels_ingredients, fp)

tsne = TSNE(n_components=2, random_state=0)
reduced_features = tsne.fit_transform(ingredient_matrix)
print("shape of reduced features ", reduced_features.shape) 
reduced_centers = tsne.fit_transform(kmeans_ingredients.cluster_centers_)

plt.figure(figsize=(12, 8))
for idx, center in enumerate(reduced_centers):
 
    cluster_points = reduced_features[ingredient_cluster_labels == idx]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {idx} ({most_common_labels_ingredients[idx]})')
    
    # If the cluster is not empty, calculate the new center and plot
    if cluster_points.size > 0:
        new_center = np.mean(cluster_points, axis=0)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5)
        
        # Plot the new center
        # plt.scatter(new_center[0], new_center[1], c='black', marker='x', s=100)
        
        # Annotate the new center with the most common label
        label = most_common_labels_ingredients[idx]
        plt.annotate(label, (new_center[0], new_center[1]), textcoords="offset points", xytext=(0,10), ha='center')


    # Annotating the center with the most common label
    # label = most_common_labels[idx]
    # plt.annotate(label, (center[0], center[1]), textcoords="offset points", xytext=(0,10), ha='center')
    

plt.title('Ingredient Clusters Visualization')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
# plt.legend()
plt.show()

######################################### end  #########################################

