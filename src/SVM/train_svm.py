import models
import torch
import joblib
from tqdm import tqdm
from sklearn import svm
from torch.utils.data import DataLoader
from sklearn.multiclass import OneVsRestClassifier
from dataset_builder import ImageDataset

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the ResNet50 model
model = models.model(pretrained=True, requires_grad=False, out_features=255).to(device)
model.eval()

# train dataset
train_dataset = ImageDataset("../../input/ingredients_classifier/images/",
                             "../../input/ingredients_classifier/train_images.txt",
                             "../../input/ingredients_classifier/train_labels.txt",
                             "../../input/ingredients_classifier/recipes.txt",
                             "../../input/ingredients_classifier/filtered_ingredients.txt",
                             True,
                             False)
# validation dataset
valid_dataset = ImageDataset("../../input/ingredients_classifier/images/",
                             "../../input/ingredients_classifier/val_images.txt",
                             "../../input/ingredients_classifier/val_labels.txt",
                             "../../input/ingredients_classifier/recipes.txt",
                             "../../input/ingredients_classifier/filtered_ingredients.txt",
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

# Train SVM model
clf = OneVsRestClassifier(svm.SVC())
clf.fit(features, labels)

# Save the model
joblib.dump(clf, '../../outputs/svm_model.pkl')
