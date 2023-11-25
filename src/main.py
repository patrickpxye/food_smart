from dataset.dataset_builder import ImageDataset
import dataset.filter_ingredients
from feature_extractor.feature_extractor import FeatureExtractor

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
    saved_model_file = '../outputs/feature_extractor.pth'
    extractor = FeatureExtractor(1095)
    extractor.train_extractor(train_dataset, valid_dataset, saved_model_file, epochs=12, lr=0.01, batch_size=32)

if __name__ == "__main__":
    main()