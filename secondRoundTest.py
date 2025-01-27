from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import pipeline
from generate_recipe_custom import model_setup,generate_recipe,ingr_to_idx
from generate_recipe import  generate_text_from_indices, generate_recipe
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM
import requests
import torch
import sqlite3
import pandas as pd
##The blue score is a metric to measure how relevant our response are 
##The rouge score is not the best metric for our model but it can help us see how many revelevant words we are retaining 
##in our responses
#We made a custom evaluators for the length and if the recipe has the correct ingredients
#There is also a function that evaluated the sentiment of the recipe, ideally we want our recipes to either be neutral or positive
    

#recipe_model = model_setup()
t5_model = T5ForConditionalGeneration.from_pretrained('./t5_recipe_generator_pretrained_model')
t5_tokenizer = T5Tokenizer.from_pretrained('./t5_recipe_generator_pretrained_model')
gpt2_tokenizer = AutoTokenizer.from_pretrained('./gpt2-recipe')
gpt2_model = AutoModelForCausalLM.from_pretrained('./gpt2-recipe')

input_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

api_url = 'https:///generate-recipe-t5'
input = {
    "ingredients": "chicken, potato"
}
#clean output, generate_recipe_t5 are all copied from the api so we can run them seamlessly in this script
#Later we will integrate api in the testing
def clean_output(output_text):
    output_text = output_text.replace("<RECIPE>", "").replace("<INGR>", "").strip()
    output_text = output_text.capitalize()

    if not output_text.endswith("."):
        output_text += ".\n"
    
    output_text = " ".join(output_text.split())
    return output_text
def generate_recipe_t5(ingredients, model, tokenizer, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = (
        f"Generate a recipe using these ingredients: {ingredients}.\n"
        f"Include preparation steps and cooking instructions in a clear, step-by-step format."
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        no_repeat_ngram_size=3,
    )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(raw_output)

def generate_output(ingredients_list): 
    """
    Generates a recipe with the given ingedients using the t5 model
    """
    try:
        ingredients = ingredients_list
        if not ingredients:
            return "Ingredients Error"

        recipe = generate_recipe_t5(ingredients, t5_model, t5_tokenizer)

        return recipe
    except requests.exceptions.RequestException as e:
        result = e

    return result

def generate_recipe_gpt2(ingredients, model, tokenizer, max_length = 512):
    input_text = f"Given these ingredients: {ingredients} generate a recipe:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=max_length,
        temperature=1.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )
    recipe = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return recipe

def get_references(ingredients): 
    """
    Goes to our original dataset and selects recipes with the given ingredients
    This create the reference material for the bleu and rouge score
    """
    where_clause = " AND ".join([f"ingredients LIKE '%{ingredient}%'" for ingredient in ingredients])
    query = f"""
    SELECT *
    FROM recipes 
    WHERE {where_clause}
    """
    conn = sqlite3.connect('13k-recipes.db')
    recipes = pd.read_sql_query(query, conn)
    conn.close()
    bleu_reference = recipes.head(4)
    rouge_reference = recipes.head(1)
    try: 
        r_ref = rouge_reference["Instructions"][0]
        b_ref = bleu_reference["Instructions"].tolist()
    except: 
        r_ref = "ERROR"
        b_ref = "ERROR"
    return r_ref, b_ref

def calculate_rouge(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores

def calculate_bleu_score(reference, prediction): 
    new_reference = []
    for r in reference: 
        new_reference.append(r.split())
    prediction_split = prediction.split()
    smoothing_f = SmoothingFunction().method1
    return sentence_bleu(new_reference, prediction_split, smoothing_function=smoothing_f)


def bleu_score_testing(reference, prediction=None): 
    score = calculate_bleu_score(reference, prediction)
    return {'BLEU score'  : score}

def rouge_score_testing(reference, prediction=None): 
    scores = calculate_rouge(reference, prediction)
    return { "ROUGE Scores": {
        "ROUGE-1:" : scores['rouge1'], 
        "ROUGE-2:" :  scores['rouge2'],
        "ROUGE-L:" : scores['rougeL']}
    }


#simple evaluation function
def evaluate_recipe(ingredients,prediction, check_1 = False):
    count = 0 
    missing = []
    used = []
    if check_1 != True:
        for i in ingredients:
            if prediction.count(i) >=1: 
                count = count +1
                used.append(i)
            else: 
                missing.append(i)
    else: 
        if prediction.count(ingredients) >=1: 
            count = count +1
    #end for loop
    percent = count / len(ingredients)
    if percent > 0.5:
        return ("Pass: Used"  + " " + str(used))
    else:
        return "Fail: did not include " + str(missing)
    
#another custom evaluation function
def evaluate_length_recipe(prediction): 
    length = len(prediction.split(" "))
    if length > 20: 
        return "Pass"
    else: 
        return "Fail: recipe is too short"

def evaluate_sentiment(recipe): 
    sentiment_model = pipeline("sentiment-analysis", 
                            model="distilbert-base-uncased-finetuned-sst-2-english", 
                            revision="714eb0f")
    if len(recipe) > 512: 
        recipe = recipe[:512]
    feedback_result = sentiment_model(inputs=recipe)
    return {"Sentiment Score" : feedback_result}


#function for running the evaluation   
def run_evaluation_ingredients(ingredients, output):
    evaluation_result = evaluate_recipe(ingredients, output)
    return {
        "Output": output,
        "Ingredient Test Result": evaluation_result
    }

def run_evaluation_length(output): 
    eval_result = evaluate_length_recipe(output)
    return {
        "Length of Output" : len(output), 
        "Length Test Result" : eval_result
    }

def output_to_string(output): 
    string_output = ""
    for o in output: 
        string_output  = string_output + " " + str(o) 
    return string_output


def test_recipe_t5(ingredients): 
    output = generate_output(ingredients)
    r_ref, b_ref = get_references(ingredients)
    results = {}
    if r_ref == "ERROR": 
        return {"error" : "Choose Different Ingredients, No Reference Material"}
    else:
        results.update(run_evaluation_ingredients(ingredients, output)) 
        results.update(run_evaluation_length(output))
        results.update(evaluate_sentiment(output)) 
        results.update(bleu_score_testing(b_ref,output)) 
        results.update(rouge_score_testing(reference=r_ref, prediction=output))
        return results
  


def test_recipe_gpt2(ingredients): 
    output = generate_recipe_gpt2(ingredients, gpt2_model, gpt2_tokenizer)
    r_ref, b_ref = get_references(ingredients)
    results = {}
    if r_ref == "ERROR": 
        return {"error" : "Choose Different Ingredients, No Reference Material"}
    else:
        results.update(run_evaluation_ingredients(ingredients, output)) 
        results.update(run_evaluation_length(output))
        results.update(evaluate_sentiment(output)) 
        results.update(bleu_score_testing(b_ref,output)) 
        results.update(rouge_score_testing(reference=r_ref, prediction=output))
        return results

##TESTING SUITE##
#Added different test cases with different numbers of ingredients
if __name__ == "__main__":
    ingredients_list = [["tomatoes", "basil", "mozzarella", "chicken"], 
                        ["chicken", "garlic", "lemon" , "salt", "rice"], 
                        ["spinach", "feta","eggs" , "butter"], 
                        ["corn", "bell peppers", "zucchini"], 
                        ["potatoes", "cheese", "sour cream"], 
                        ["beef", "carrots", "potatoes", "thyme"], 
                        ["chicken", "broccoli", "rice", "soy sauce"], 
                        ["salmon", "asparagus", "lemon", "dill"], 
                        ["tofu", "broccoli", "scallions", "pepper"], 
                        ["pork", "onions", "sage", "butter"]]
    
    #iterates through the test cases
    def create_file(file_name, input): 
        count = 0
        with open(file_name, 'w',encoding="utf-8") as file:
            for case in input:
                count = count+1
                file.write(f"Test Case #" + str(count) + "\n")
                file.write(f"{'Output':<15}: {case['Output']}\n")
                file.write(f"{'Ingredient Test Result':<25}: {case['Ingredient Test Result']}\n")
                file.write(f"{'Length of Output':<20}: {case['Length of Output']}\n")
                file.write(f"{'Length Test Result':<20}: {case['Length Test Result']}\n")
                file.write(f"{'Sentiment Score':<15}: {case['Sentiment Score']}\n")
                file.write(f"{'BLEU Score':<15}: {case['BLEU score']:.4f}\n")
                
                for score_type, scores in case['ROUGE Scores'].items():
                    file.write(f"{score_type:<15}: Precision={scores[0]:.4f}, Recall={scores[1]:.4f}, F-Measure={scores[2]:.4f}\n")
                
                file.write("\n" + "-"*50 + "\n")
        print(f"Test case output written to {file_name}")

    t5_test = []
    gpt2_test = []
    for i in range(0, len(ingredients_list)): 
        #start with t5 model
        t5_vals = test_recipe_t5(ingredients_list[i])
        t5_test.append(t5_vals)

        #then we perform the same test with the gpt2 model
        gpt2_vals = test_recipe_gpt2(ingredients_list[i])
        gpt2_test.append(gpt2_vals)

    create_file("t5_test.txt", t5_test)
    create_file("gpt2_test.txt", gpt2_test)

        