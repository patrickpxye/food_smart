import torch
import joblib
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.dataset_builder import ImageDataset
from feature_extractor.resnet50_feature_extractor import Resnet50FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from feature_extractor.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from feature_extractor.vgg16_feature_extractor import VGG16FeatureExtractor
from feature_extractor.inceptionv3_feature_extractor import InceptionV3FeatureExtractor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the ResNet50 model
# filtered: 255, unfiltered: 1095
extractor = Resnet50FeatureExtractor(1095).load_extractor('../../feature_extractor/resnet50_feature_extractor.pth')
#extractor = EfficientNetFeatureExtractor(1095).load_extractor('../../feature_extractor/efficientNet_feature_extractor.pth')
#extractor = VGG16FeatureExtractor(1095).load_extractor('../../feature_extractor/vgg16_feature_extractor.pth')
#extractor = InceptionV3FeatureExtractor(1095).load_extractor('../../feature_extractor/InceptronV3_feature_extractor.pth')

# train dataset
train_dataset = ImageDataset("../../input/ingredients_classifier/images/",
                             "../../input/ingredients_classifier/train_images.txt",
                             "../../input/ingredients_classifier/train_labels.txt",
                             "../../input/ingredients_classifier/recipes.txt",
                             "../../input/ingredients_classifier/ingredients.txt",
                             True,
                             False)
# validation dataset
valid_dataset = ImageDataset("../../input/ingredients_classifier/images/",
                             "../../input/ingredients_classifier/val_images.txt",
                             "../../input/ingredients_classifier/val_labels.txt",
                             "../../input/ingredients_classifier/recipes.txt",
                             "../../input/ingredients_classifier/ingredients.txt",
                             False,
                             False)
batch_size = 32
# train data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False
)

features = []
labels = []

for i, data in tqdm(enumerate(train_loader), total=int(len(train_dataset) / train_loader.batch_size)):
    train_data, target = data['image'].to(device), data['label'].to(device)
    outputs = extractor(train_data)
    features.extend(outputs.detach().cpu().numpy())
    labels.extend(target.detach().cpu().numpy())

# Train ensemble model
features = np.array(features)
labels = np.array(labels)

validation_features = []
validation_labels = []

for i, data in tqdm(enumerate(valid_loader), total=int(len(valid_loader) / valid_loader.batch_size)):
    train_data, target = data['image'].to(device), data['label'].to(device)
    outputs = extractor(train_data)
    validation_features.extend(outputs.detach().cpu().numpy())
    validation_labels.extend(target.detach().cpu().numpy())

validation_features = np.array(validation_features)
validation_labels = np.array(validation_labels)

# Train ensemble model
features = np.array(features)
labels = np.array(labels)

accuracies = []
epochs = []
num_epochs = 100
clf = RandomForestClassifier(n_estimators=10, random_state=42, warm_start=True)
multi_target_forest = MultiOutputClassifier(clf, n_jobs=-1)
previous_accuracy = 0
accuracy = 1
epoch = 0
while abs(previous_accuracy-accuracy) > 1e-5:
    # Fit your model
    multi_target_forest.fit(features, labels)

    # Increase the number of estimators for the next epoch
    clf.n_estimators += 10

    # Predict on your validation set (or test set)
    predictions = multi_target_forest.predict(validation_features)

    # Calculate accuracy
    accuracy = accuracy_score(validation_labels, predictions)

    # Append accuracy and epoch to lists
    accuracies.append(accuracy)
    epochs.append(epoch)

    print(f"Epoch: {epoch}, Accuracy: {accuracy}")
    epoch += 1
    previous_accuracy = accuracy

#multi_target_forest.fit(features, labels)
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.show()

# Save the model
joblib.dump(multi_target_forest, '../../outputs/resnet50_ensemble_model.pkl')
