import feature_extractor
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from torch.utils.data import DataLoader
from sklearn.multiclass import OneVsRestClassifier
from dataset_builder import ImageDataset
from feature_extractor.feature_extractor import FeatureExtractor

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the feature extractor
extractor = FeatureExtractor(1095).load_extractor('../outputs/feature_extractor.pth')

# load SVM model
clf = joblib.load('../../outputs/svm_model.pkl')

# prepare the test dataset and dataloader
test_data = ImageDataset("../../input/ingredients_classifier/images/",
                         "../../input/ingredients_classifier/test_images.txt",
                         "../../input/ingredients_classifier/test_labels.txt",
                         "../../input/ingredients_classifier/recipes.txt",
                         "../../input/ingredients_classifier/filtered_ingredients.txt",
                         False,
                         True)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False
)

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