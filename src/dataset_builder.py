import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    base_dir = "../input/ingredients_classifier/images/"
    def __init__(self, image_file, label_file, recipe_file, ingredient_file, for_training, for_test):
        self.image_file = image_file
        self.label_file = label_file
        self.recipe_file = recipe_file
        self.ingredient_file = ingredient_file
        self.for_training = for_training
        self.for_test = for_test
        self.image_transform = None

        self.ingredient_index, _ = self.build_ingredient_indices()
        self.label_matrix = self.build_label_matrix(self.ingredient_index)
        self.image_index_to_recipe_index, self.image_index_to_name = self.build_image_recipe_map()
        self.build_image_transformer()

    def __len__(self):
        return len(self.image_index_to_name)

    def __getitem__(self, index):
        image = cv2.imread(f"../input/ingredients_classifier/images/{self.image_index_to_name[index]}")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.image_transform(image)
        targets = self.label_matrix[self.image_index_to_recipe_index[index]]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }

    def build_ingredient_indices(self):
        with open(self.ingredient_file, 'r') as f:
            ingredients = f.read().split(',')
        ingredient_to_index = {ingredient: index for index, ingredient in enumerate(ingredients)}
        index_to_ingredient = {v: k for k, v in ingredient_to_index.items()}
        return ingredient_to_index, index_to_ingredient

    def build_label_matrix(self, ingredient_to_index):
        with open(self.recipe_file, 'r') as f:
            recipes = [line.strip().split(',') for line in f]

        # Initialize a matrix of zeros with dimensions (number of recipes x number of ingredients)
        label_matrix = np.zeros((len(recipes), len(ingredient_to_index)))
        for i, recipe in enumerate(recipes):
            for ingredient in recipe:
                # If the ingredient is in the ingredient_to_index map, set the corresponding cell to 1
                if ingredient in ingredient_to_index:
                    j = ingredient_to_index[ingredient]
                    label_matrix[i, j] = 1
        return label_matrix

    def build_image_recipe_map(self):
        with open(self.image_file, 'r') as f:
            image_names = f.read().splitlines()
        with open(self.label_file, 'r') as f:
            recipe_indices = [int(line.strip()) for line in f]

        # Create a map from image URL index to recipe index
        image_index_to_recipe_index = {i: recipe_indices[i] for i in range(len(image_names))}

        # Create a map from image URL to its index
        image_index_to_name = {i: name for i, name in enumerate(image_names)}

        return image_index_to_recipe_index, image_index_to_name

    def build_image_transformer(self):
        # set the training data images and labels
        if self.for_training == True:
            # define the training transforms
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        # set the validation data images and labels
        elif self.for_training == False and self.for_test == False:
            # define the validation transforms
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
            ])
        elif self.for_test == True and self.for_training == False:
            # define the test transforms
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])



