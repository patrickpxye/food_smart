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

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the ResNet50 model
# filtered: 255, unfiltered: 1095
#extractor = Resnet50FeatureExtractor(1095).load_extractor('../../feature_extractor/efficientNet_feature_extractor.pth')
#extractor = EfficientNetFeatureExtractor(1095).load_extractor('../../feature_extractor/efficientNet_feature_extractor.pth')
#extractor = VGG16FeatureExtractor(1095).load_extractor('../../feature_extractor/vgg16_feature_extractor.pth')
extractor = InceptionV3FeatureExtractor(1095).load_extractor('../../feature_extractor/InceptronV3_feature_extractor.pth')

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
clf = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_forest = MultiOutputClassifier(clf, n_jobs=-1)
multi_target_forest.fit(features, labels)

# Save the model
joblib.dump(multi_target_forest, '../../outputs/inceptionv3_ensemble_model.pkl')
