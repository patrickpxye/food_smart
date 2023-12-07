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
# from efficientnet_pytorch import EfficientNet
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
                       False, 
                       )

# print("train data len", train_data.index_to_ingredient.__len__())
# 255
# print("train data len", train_data.index_to_ingredient)

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

# # Load the pretrained EfficientNet-B7 model
# efficientnet_model = EfficientNet.from_pretrained('efficientnet-b7')

# # Freeze all the layers
# for param in efficientnet_model.parameters():
#     param.requires_grad = False

# # Modify the model to remove the last classification layer
# # EfficientNet models have their classifier in '_fc', so we set it to an identity layer
# efficientnet_model._fc = torch.nn.Identity()

# # Set the model to evaluation mode
# efficientnet_model.eval()

# Load the saved K-Means model
kmeans = load('saved_outputs_multi/kmeans_model.joblib')

# Load the saved cluster labels
saved_cluster_labels = np.load('saved_outputs_multi/cluster_labels.npy')

# Later, to load the dictionary
with open('saved_outputs_multi/most_common_labels.pkl', 'rb') as fp:
    most_common_labels = pickle.load(fp)
    
# Extract the label matrix (ingredients matrix)
ingredient_matrix = np.array([train_data[i]['label'] for i in range(len(train_data))])

# Load the saved K-Means model
kmeans_ingredients = load('saved_outputs_multi/kmeans_model_ingredients.joblib')

# Load the saved cluster labels
saved_cluster_labels = np.load('saved_outputs_multi/cluster_labels_ingredients.npy')


with open('saved_outputs_multi/most_common_labels_ingredients.pkl', 'rb') as fp:
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
    scores = []
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
        
        actual_set = set(actual) 
        count = sum(pred in actual_set for pred in predicted)
        score = count/ max(len(actual), 1)
        scores.append(score)
               

        # Debugging print
        # print(f"Actual: {actual}, Predicted: {predicted}")
        print(f"F1: {f1}, Precision: {precision}, Recall: {recall}, Score: {score}")

    avg_f1 = sum(f1_scores) /max(len(f1_scores),1)
    avg_precision = sum(precisions) / max(len(precisions),1)
    avg_recall = sum(recalls) / max(len(recalls), 1)
    avg_score = sum(scores) / max(len(scores), 1)
    
    return avg_f1, avg_precision, avg_recall, avg_score

actual_labels = []
predicted_labels = []
top_n_predicted_labels = []

counter_predicted_label = 0
total_plates_that_we_can_predict = 0
# print("most common labels", most_common_labels) # image cluster lables
# print("predicted clusters", predicted_clusters)
# print("ingredient cluster labels", ingredient_cluster_labels) # ingredient cluster lables
unique_ingredient_cluster_labels = []
# Convert values of ingredient_cluster_labels to a set for faster lookup
# labels = set(most_common_labels.values())
labels = set(np.unique(np.concatenate(list(most_common_labels.values()))))

ingredient_labels = set(np.unique(np.concatenate(list(ingredient_cluster_labels.values()))))
# Iterate through the items in most_common_labels
for item in ingredient_labels:
    # If the item is in the set of labels
    if item not in labels:
        unique_ingredient_cluster_labels.append(item)
# print("unique ingredient cluster labels", unique_ingredient_cluster_labels)
print("unique ingredient cluster labels length",len(unique_ingredient_cluster_labels))

# in the multi label case there is only 1 label that is not accounted for

for counter, data in enumerate(valid_loader):
    images, targets, label_name = data['image'].to(device), data['label'], data['label_name']
    # print("images", range(images.size(0)))
    # Loop through each image in the batch
    for i in range(images.size(0)):
        image = images[i].squeeze().detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))

        predicted_label_names = most_common_labels[predicted_clusters[counter * images.size(0) + i]]
        # print("predicted label name", predicted_label_names)
        # print("actual label name", label_name[i])
        
        predicted_in_ingredient_label = False
        for item in predicted_label_names:
            if item in ingredient_labels:
                predicted_in_ingredient_label = True
                break
        # only counting if we can predict the image cluster and the image cluster is in the ingredient cluster
        if label_name[i] in predicted_label_names  and predicted_in_ingredient_label:
            counter_predicted_label += 1
        # only add and calculate the metrics if we can predict the image cluster and the image cluster is in the ingredient cluster
        if predicted_in_ingredient_label:
            total_plates_that_we_can_predict += 1
            # indices for the clusters of  kmeans_ingredients     
            # unique_cluster_indices = [index for index, item in  ingredient_cluster_labels.items() if item in predicted_label_names]
            unique_cluster_indices = [index for index, item_arr in ingredient_cluster_labels.items() if any(predicted_item in item_arr for predicted_item in predicted_label_names)]

            # print("unique cluster indices", unique_cluster_indices)
            # Aggregate all ingredients from these clusters
            
            all_ingredients = set()
            all_ingredients_arr = []

            for index in unique_cluster_indices:
                # Get the labels assigned by the KMeans model
                labels = kmeans_ingredients.labels_

                # Get the indices of items that belong to cluster 'index'
                indices_in_cluster = np.where(labels == index)[0]
                # print("indices in cluster", indices_in_cluster)

                # Retrieve the ingredient names belonging to this cluster
                for idx in indices_in_cluster:
                    if idx <= train_data.__len__():
                        ingredient_idx_vector = train_data.__getitem__(idx)['label']
                        ingredient_idx = np.where(ingredient_idx_vector == 1)[0]
                        for  y in ingredient_idx:
                            if y <= train_data.index_to_ingredient.__len__():
                                all_ingredients.add(train_data.index_to_ingredient[y])
                                all_ingredients_arr.append(train_data.index_to_ingredient[y])
                            else :
                                print(f"Index {y} not in range of index data.")
                                continue
                    else:
                        print(f"Index {idx} not in range of train data.")
                        continue
            ing_counts = Counter(all_ingredients_arr)
    
            # Get the top n most common labels
            # The most_common method returns a list of tuples (label, count)
            top_n = ing_counts.most_common(7)

            # Extract just the labels from the tuples
            top_n_ing = [label for label, count in top_n]
            # Convert the set of all ingredients to a list and then to a string
            predicted_ingredients = '    '.join(list(top_n_ing))
            print("Predicted Ingredients:", predicted_ingredients)
            target_indices = np.where(targets[i] == 1)[0]
            actual_label_names = [valid_data.index_to_ingredient[idx] for idx in target_indices]
            print("Actual Ingredients:", actual_label_names)

            # Plotting the image
            # plt.imshow(image)
            # plt.axis('off')
            # string_predicted = f"{top_n_ing} "
            # string_actual = '    '.join(actual_label_names)
            # plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
            # # plt.savefig(f"outputs_k_means_multi/inference_{counter}_{i}.jpg")
            # plt.close()
            
            # only counting if we can predict the image cluster and the image cluster is in the ingredient cluster
            
            actual_labels.append(set(actual_label_names))
            predicted_labels.append(all_ingredients)
            top_n_predicted_labels.append(top_n_ing)
            
# After the loop
average_f1_all_pre, average_precision_all_pre, average_recall_all_pre, average_score_all_pre = compute_metrics(actual_labels, predicted_labels)
average_f1_top_pre, average_precision_top_pre, average_recall_top_pre, average_score_top_pre = compute_metrics(actual_labels, top_n_predicted_labels)
# print(f"Number of correct labels: {counter_predicted_label/valid_data.__len__()}")
print(f"Total number of plates that we can predict: {total_plates_that_we_can_predict}, Total number of plates: {valid_data.__len__()}")
print(f"Number of correct labels: {counter_predicted_label/total_plates_that_we_can_predict}")
print(f"Average F1 Score all Predictions: {average_f1_all_pre}")
print(f"Average Precision all Predictions: {average_precision_all_pre}")
print(f"Average Recall all Predictions: {average_recall_all_pre}")
print(f"Average Score all Predictions: {average_score_all_pre}")

print(f"Average F1 Score top Predictions: {average_f1_top_pre}")
print(f"Average Precision top Predictions: {average_precision_top_pre}")
print(f"Average Recall top Predictions: {average_recall_top_pre}")
print(f"Average Score top Predictions: {average_score_top_pre}")

# 5 image, 5 ingredient clusters, top 5 ingredients
# Total number of plates that we can predict: 634, Total number of plates: 634
# Number of correct labels: 0.055205047318611984
# Average F1 Score all Predictions: 0.034984704318865634
# Average Precision all Predictions: 0.034984704318865634
# Average Recall all Predictions: 0.034984704318865634
# Average Score all Predictions: 0.971895546619522
# Average F1 Score top Predictions: 0.17305936976260555
# Average Precision top Predictions: 0.17305936976260555
# Average Recall top Predictions: 0.17305936976260555
# Average Score top Predictions: 0.24296245754735785