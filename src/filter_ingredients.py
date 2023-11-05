from collections import Counter

def filter(ingredient_file, recipe_file, save_file):
    # Read ingredients from file
    with open(ingredient_file, 'r') as f:
        ingredients = f.read().split(',')
    # Read recipes from file
    with open(recipe_file, 'r') as f:
        recipes = f.read().split('\n')
    # Count occurrences of each ingredient in the recipes
    ingredient_counts = Counter()
    for recipe in recipes:
        for ingredient in recipe.split(','):
            if ingredient in ingredients:
                ingredient_counts[ingredient] += 1
    # Filter out ingredients that appear in less than five recipes
    filtered_ingredients = {ingredient: count for ingredient, count in ingredient_counts.items() if count >= 5}

    with open(save_file, 'w') as f:
        f.write(','.join(filtered_ingredients))