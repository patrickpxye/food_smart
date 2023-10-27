from dataset_builder import ImageDataset

def main():

    dataset = ImageDataset("../input/ingredients_classifier/train_images.txt",
                           "../input/ingredients_classifier/train_labels.txt",
                           "../input/ingredients_classifier/recipes.txt",
                           "../input/ingredients_classifier/ingredients.txt",
                           True,
                           False)

    print(f'Ingredient vacabulary size : {len(dataset.ingredient_index)}')

if __name__ == "__main__":
    main()