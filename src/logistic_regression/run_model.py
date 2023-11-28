import models
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset_builder import ImageDataset
from torch.utils.data import DataLoader

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = models.model(requires_grad=False, out_features=1095).to(device)
# load the model checkpoint
checkpoint = torch.load('../../outputs/logistic_model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    # get the predictions by passing the image through the model
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
    best = sorted_indices[-5:]
    string_predicted = ''
    string_actual = ''
    for i in range(len(best)):
        string_predicted += f"{test_data.index_to_ingredient[best[i].item()]}    "
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