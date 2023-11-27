from feature_extractor.feature_extractor import FeatureExtractor
import torch
import joblib
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.multiclass import OneVsRestClassifier
from dataset.dataset_builder import ImageDataset
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the ResNet50 model
# filtered: 255, unfiltered: 1095
extractor = FeatureExtractor(1095).load_extractor('../../outputs/feature_extractor.pth')

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

# Train SVM model
features = np.array(features)
labels = np.array(labels)
clf = OneVsRestClassifier(LinearSVC(penalty='l1', C=1, dual=False))

'''
parameters = {
    #'vect__ngram_range': [(1, 3), (1, 4)],
    #'vect__max_df': [1, 0.8],
#     'vect__min_df': [1, 0.8, 0],
    #'vect__use_idf': [True],
    'estimator__penalty': ['l1'],
    'estimator__C': [1, 10, 100],
    'estimator__dual': [False],
}

gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
gs_clf.fit(features, labels)

valid_features = []
valid_labels = []
for i, data in tqdm(enumerate(valid_loader), total=int(len(valid_dataset) / valid_loader.batch_size)):
    valid_data, target = data['image'].to(device), data['label'].to(device)
    outputs = extractor(valid_data)
    valid_features.extend(outputs.detach().cpu().numpy())
    valid_labels.extend(target.detach().cpu().numpy())

Y_predicted = gs_clf.predict(valid_features)
print(metrics.classification_report(valid_labels, Y_predicted))
print(gs_clf.best_params_)
print(gs_clf.best_score_)
'''
# Save the model
joblib.dump(clf, '../../outputs/svm_model.pkl')
