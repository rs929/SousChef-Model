import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import spacy


output_dir = "./t5_recipe_generator_pretrained_model"
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_md")

loaded_model = T5ForConditionalGeneration.from_pretrained(output_dir)
loaded_tokenizer = T5Tokenizer.from_pretrained(output_dir)

def validate_action_ingredient_contextual(action, ingredient, threshold=0.5):
    """
    Validates the contextual similarity between an action and an ingredient 
    using a pre-trained SentenceTransformer model.

    Parameters:
      action (str): The action text (e.g., "chop onions").
      ingredient (str): The ingredient text (e.g., "onion").
      threshold (float): The similarity threshold above which the action and ingredient are considered valid. Defaults to 0.5.

    Returns:
      bool: True if the similarity score exceeds the threshold, False otherwise.
    """
    action_embedding = bert_model.encode(action)
    ingredient_embedding = bert_model.encode(ingredient)
    similarity = util.cos_sim(action_embedding, ingredient_embedding).item()
    return similarity > threshold

def validate_action_ingredient_spacy(action, ingredient, threshold=0.5):
    """
    Validates the similarity between an action and an ingredient 
    using spaCy's word vector similarity.

    Parameters:
      action (str): The action text (e.g., "chop onions").
      ingredient (str): The ingredient text (e.g., "onion").
      threshold (float): The similarity threshold above which the action and ingredient are considered valid. Defaults to 0.5.

    Returns:
      bool: True if the similarity score exceeds the threshold, False otherwise.
    """
    similarity = nlp(action).similarity(nlp(ingredient))
    return similarity > threshold

def validate_word_groupings(recipe, ingredients, n=2, threshold=0.5):
    """
    Validates n-word groupings in a recipe against a list of ingredients 
    using both contextual (BERT) and lexical (spaCy) similarity.

    Parameters:
      recipe (str): The generated recipe text to validate.
      ingredients (list of str): A list of ingredient names.
      n (int): The number of words in each grouping to validate. Defaults to 2.
      threshold (float): The similarity threshold above which a grouping is considered valid. Defaults to 0.5.

    Returns:
      - int: The number of invalid groupings found in the recipe.
      - list of str: The list of invalid word groupings.
    """
    words = recipe.split()
    invalid_groupings = []

    for i in range(len(words) - n + 1):
        word_group = " ".join(words[i : i + n])
        valid = any(
            validate_action_ingredient_contextual(word_group, ingredient, threshold)
            or validate_action_ingredient_spacy(word_group, ingredient, threshold)
            for ingredient in ingredients
        )
        if not valid:
            invalid_groupings.append(word_group)

    return len(invalid_groupings), invalid_groupings


def find_best_recipe(ingredients_list, model, tokenizer, num_recipes=10, n=2, threshold=0.5):
    """
    Generates multiple recipes using a T5 model and returns the recipe 
    with the least invalid word groupings.

    Parameters:
      ingredients_list (list of str): The list of ingredients for the recipe.
      model (transformers.PreTrainedModel): The T5 model for recipe generation.
      tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the T5 model.
      num_recipes (int): The number of recipes to generate. Defaults to 10.
      n (int): The n-gram size for validating word groupings. Defaults to 2.
      threshold (float): The similarity threshold for validation. Defaults to 0.5.

    Returns:
      str: The best recipe (i.e., the one with the least invalid groupings).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    formatted_ingredients = ", ".join(ingredients_list)
    input_text = (
        f"Generate a recipe using these ingredients: {formatted_ingredients}.\n"
        f"Include preparation steps and cooking instructions in a clear, step-by-step format."
    )

    best_recipe = None
    least_invalid_count = float('inf')
    best_invalid_groupings = []

    for i in range(num_recipes):
        inputs = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        outputs = model.generate(
            inputs["input_ids"],
            max_length=200000,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=3,
        )

        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe = clean_output(raw_output)

        invalid_count, invalid_groupings = validate_word_groupings(recipe, ingredients_list, n, threshold)

        if invalid_count < least_invalid_count:
            best_recipe = recipe
            least_invalid_count = invalid_count
            best_invalid_groupings = invalid_groupings

        print(f"Generated Recipe {i + 1}/{num_recipes}")
        print(recipe)
        print()

    print(f"Best Recipe Invalid Count: {least_invalid_count}")
    print(f"Best Recipe Invalid Groupings: {', '.join(best_invalid_groupings)}\n")
    return best_recipe

def clean_output(output_text):
    """
    Cleans and formats the generated recipe text by removing unwanted tokens 
    and ensuring proper capitalization and punctuation.

    Args:
      output_text (str): The raw output text from the T5 model.

    Returns:
      str: The cleaned and formatted recipe text.
    """
    output_text = output_text.replace("<RECIPE>", "").replace("<INGR>", "").strip()
    output_text = output_text.capitalize()

    if not output_text.endswith("."):
        output_text += ".\n"
    
    output_text = " ".join(output_text.split())
    return output_text

def get_user_input():
    ingredient_list = input("Enter the list of ingredients (comma-separated): ")
    n_gram_size = input("Enter the desired n-gram size: ")
    num_recipes = input("Enter the number of recipe iterations: ")

    if not n_gram_size.isdigit():
        print("Error: n-gram size must be an integer.")
        return None

    if not num_recipes.isdigit():
        print("Error: Number of recipe iterations must be an integer.")
        return None

    ingredient_list = [ingredient.strip() for ingredient in ingredient_list.split(',')]
    n_gram_size = int(n_gram_size)
    num_recipes = int(num_recipes)

    print(find_best_recipe(ingredient_list, loaded_model, loaded_tokenizer, num_recipes, n_gram_size))


if __name__ == "__main__":
    get_user_input()