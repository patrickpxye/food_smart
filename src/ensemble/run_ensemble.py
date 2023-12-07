import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.dataset_builder import ImageDataset
from feature_extractor.resnet50_feature_extractor import Resnet50FeatureExtractor
from sklearn.metrics import f1_score, precision_score, recall_score
from feature_extractor.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from feature_extractor.vgg16_feature_extractor import VGG16FeatureExtractor
from feature_extractor.inceptionv3_feature_extractor import InceptionV3FeatureExtractor

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
#extractor = Resnet50FeatureExtractor(1095).load_extractor('../../feature_extractor/efficientNet_feature_extractor.pth')
# extractor = EfficientNetFeatureExtractor(1095).load_extractor('../../feature_extractor/efficientNet_feature_extractor.pth')
#extractor = VGG16FeatureExtractor(1095).load_extractor('../../feature_extractor/vgg16_feature_extractor.pth')
extractor = InceptionV3FeatureExtractor(1095).load_extractor('../../feature_extractor/InceptronV3_feature_extractor.pth')

# load SVM model
clf = joblib.load('../../outputs/inceptionv3_ensemble_model.pkl')

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

def evaluate_ensemble(extractor, clf, test_loader, device, num_batches=1):
    val_features = []
    val_labels = []
    batch_count = 0

    with torch.no_grad():
        for data in test_loader:
            if num_batches is not None and batch_count >= num_batches:
                break
            inputs, labels = data['image'].to(device), data['label'].to(device)
            features = extractor(inputs).detach().cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.detach().cpu().numpy())
            batch_count += 1

    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    predictions = clf.predict(val_features)

    f1_micro = f1_score(val_labels, predictions, average='micro')
    precision_score_micro = precision_score(val_labels, predictions, average='micro')
    recall_score_micro = recall_score(val_labels, predictions, average='micro')

    print(f"F1 Score (Micro): {f1_micro}")
    print(f"Precision (Micro): {precision_score_micro}")
    print(f"Recall (Micro): {recall_score_micro}")

    return f1_micro, precision_score_micro, recall_score_micro

f1, precision, recall = evaluate_ensemble(extractor, clf, test_loader, device, 100)


for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    # get the predictions by passing the image through the model
    outputs = extractor(image)
    # outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    predictions = clf.predict(outputs)
    print(f'prediction probability {predictions}')
    sorted_indices = np.argsort(predictions[0])
    print(f'sorted indices {sorted_indices}')
    best = sorted_indices[-5:]
    string_predicted = ''
    string_actual = ''
    for i in range(len(best)):
        if best[i].item() in test_data.index_to_ingredient:
            string_predicted += f"{test_data.index_to_ingredient[best[i].item()]}    "
        else:
            print(f'key {best[i].item()} not found in dictionary')
    for i in range(len(target_indices)):
        string_actual += f"{test_data.index_to_ingredient[target_indices[i]]}    "
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(f"../../outputs/inference_{counter}.jpg")
    # plt.show()