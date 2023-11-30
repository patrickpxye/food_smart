from dataset.dataset_builder import ImageDataset

from feature_extractor.resnet50_feature_extractor import Resnet50FeatureExtractor

# train dataset
train_data = ImageDataset("../../input/ingredients_classifier/images/",
                          "../../input/ingredients_classifier/train_images.txt",
                          "../../input/ingredients_classifier/train_labels.txt",
                          "../../input/ingredients_classifier/recipes.txt",
                          "../../input/ingredients_classifier/ingredients.txt",
                          True,
                          False)
# validation dataset
valid_data = ImageDataset("../../input/ingredients_classifier/images/",
                          "../../input/ingredients_classifier/val_images.txt",
                          "../../input/ingredients_classifier/val_labels.txt",
                          "../../input/ingredients_classifier/recipes.txt",
                          "../../input/ingredients_classifier/ingredients.txt",
                          False,
                          False)

extractor = Resnet50FeatureExtractor(1095)

saved_model_file = f'../outputs/{extractor.name}_feature_extractor.pth'
extractor.train_extractor(train_data, valid_data, saved_model_file, epochs=12, lr=0.01, batch_size=32)

