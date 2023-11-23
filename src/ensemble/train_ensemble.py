import models
import torch
import joblib
from tqdm import tqdm
from sklearn import svm
from torch.utils.data import DataLoader
from sklearn.multiclass import OneVsRestClassifier
from dataset_builder import ImageDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multioutput import ClassifierChain

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the ResNet50 model
# filtered: 255, unfiltered: 1095
model = models.model(requires_grad=False, out_features=1095).to(device)
model.eval()

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
    outputs = model(train_data)
    features.extend(outputs.detach().cpu().numpy())
    labels.extend(target.detach().cpu().numpy())

# Train ensemble model
features = np.array(features)
labels = np.array(labels)
clf1 = OneVsRestClassifier(svm.SVC())
clf2 = OneVsRestClassifier(LogisticRegression())
clf3 = OneVsRestClassifier(DecisionTreeClassifier())
ecf = VotingClassifier(estimators=[('svc', clf1), ('lr', clf2), ('dt', clf3)], voting='hard')
chain = ClassifierChain(ecf)
chain.fit(features, labels)

# Save the model
joblib.dump(chain, '../../outputs/ensemble_model.pkl')
