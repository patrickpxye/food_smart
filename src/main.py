from dataset.dataset_builder import ImageDataset
from feature_extractor.resnet50_feature_extractor import Resnet50FeatureExtractor
from feature_extractor.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from feature_extractor.inceptionv3_feature_extractor import InceptionV3FeatureExtractor
from feature_extractor.vgg16_feature_extractor import VGG16FeatureExtractor
def main():

    '''
    dataset = ImageDataset("../input/ingredients_classifier/train_images.txt",
                           "../input/ingredients_classifier/train_labels.txt",
                           "../input/ingredients_classifier/recipes.txt",
                           "../input/ingredients_classifier/ingredients.txt",
                           True,
                           False)

    print(f'Ingredient vacabulary size : {len(dataset.ingredient_index)}')


    filter_ingredients.filter("../input/ingredients_classifier/ingredients.txt",
                                 "../input/ingredients_classifier/recipes.txt",
                                  "../input/ingredients_classifier/filtered_ingredients.txt")
    '''

    # train dataset
    train_dataset = ImageDataset("../input/ingredients_classifier/images/",
                                 "../input/ingredients_classifier/train_images.txt",
                                 "../input/ingredients_classifier/train_labels.txt",
                                 "../input/ingredients_classifier/recipes.txt",
                                 "../input/ingredients_classifier/ingredients.txt",
                                 True,
                                 False)
    # validation dataset
    valid_dataset = ImageDataset("../input/ingredients_classifier/images/",
                                 "../input/ingredients_classifier/val_images.txt",
                                 "../input/ingredients_classifier/val_labels.txt",
                                 "../input/ingredients_classifier/recipes.txt",
                                 "../input/ingredients_classifier/ingredients.txt",
                                 False,
                                 False)

    # extractor = Resnet50FeatureExtractor(1095)
    extractor = EfficientNetFeatureExtractor(1095)
    # extractor = InceptionV3FeatureExtractor(1095)
    # extractor = VGG16FeatureExtractor(1095)

    saved_model_file = f'../outputs/{extractor.name}_feature_extractor.pth'
    extractor.train_extractor(train_dataset, valid_dataset, saved_model_file, epochs=12, lr=0.01, batch_size=32)

if __name__ == "__main__":
    main()