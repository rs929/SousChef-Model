from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM
import os
import json
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import base64
from io import BytesIO


app = Flask(__name__)
CORS(app)

# Here, we load the pretrained t5 model
t5_model = T5ForConditionalGeneration.from_pretrained('./t5_recipe_generator_pretrained_model')
t5_tokenizer = T5Tokenizer.from_pretrained('./t5_recipe_generator_pretrained_model')

# Here, we load the pretrained gpt2 model
gpt2_tokenizer = AutoTokenizer.from_pretrained('./gpt2-recipe')
gpt2_model = AutoModelForCausalLM.from_pretrained('./gpt2-recipe')

# Here, we load the pretrained object detction model and set it up
image_model = models.resnet18(pretrained=True)
num_classes = 498

image_model.fc = nn.Linear(image_model.fc.in_features, num_classes)
try:
    base_dir = os.path.dirname(__file__) 
except NameError:
    base_dir = os.getcwd() 
model_path = os.path.join(base_dir, "../Image_Model/resnet18_images.pth") 

image_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
image_model.eval()
with open("./../Image_Model/encodings.json", "r") as f:
    label_to_index = json.load(f)

index_to_label = {v: k for k, v in label_to_index.items()}

"""
Given an image, returns the predicted label by the object detection model.

Args:
    img (image file): image of a food item/ingredient.

Returns:
    str: The predicted label of the image.
"""
def generate_image_label(img): 
    image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_image = image_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = image_model(transformed_image)  
        predicted_index = torch.argmax(outputs, dim=1).item()  
        predicted_label = index_to_label[predicted_index]  
    print("Predicted Label: ", predicted_label )
    return predicted_label
    

"""
Generates a recipe using a pretrained t5 model, given a list of ingredients

Args:
    ingredients (str): A comma separeated string of ingredients.
    model: The pretrained t5 model.
    tokenizer: The pretrained t5 tokenizer.
    max_length (int): The max token length for t5 (make sure to leave as default to not cause errors!).

Returns:
    str: A string representing the recipe the t5 model constructs using the given ingredients.
"""
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

"""
A function to clean recipe output so that it is easier read by end users.  Removes special
tokens such as <RECIPE> and capitalizes necessary text.

Args:
    output_text (str): A string recipe output from a recipe generation function.

Returns:
    str: The string recipe passed in as an argument, with special tokens removed and capitalization
    applied
"""
def clean_output(output_text):
    output_text = output_text.replace("<RECIPE>", "").replace("<INGR>", "").strip()
    output_text = output_text.capitalize()

    if not output_text.endswith("."):
        output_text += ".\n"
    
    output_text = " ".join(output_text.split())
    return output_text

"""
Generates a recipe using a pretrained gpt2 model, given a list of ingredients

Args:
    ingredients (str): A comma separeated string of ingredients.
    model: The pretrained gpt2 model.
    tokenizer: The pretrained gpt2 tokenizer.
    max_length (int): The max token length for gpt2 (make sure to leave as default to not cause errors!).

Returns:
    str: A string representing the recipe the gpt2 model constructs using the given ingredients.
"""
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

"""
_____________________ BEGIN API ENDPOINTS _____________________________
"""

"""
Endpoint to generate a recipe from input ingredients, using the pretrained t5
model.

Method: POST

Request Body (JSON):
    {
        "ingredients" : <string of ingredients, separated by commas>
    }

Returns (JSON):
    {
        "recipe" : <model generated recipe string>
    }
"""
@app.route('/generate-recipe-t5', methods=['POST'])
def generate_recipe_endpoint():
    data = request.json
    ingredients = data.get('ingredients')

    if not ingredients:
        return jsonify({"error": "Ingredients are required"}), 400

    recipe = generate_recipe_t5(ingredients, t5_model, t5_tokenizer)

    return jsonify({"recipe": recipe})

"""
Endpoint to generate a recipe from input ingredients, using the pretrained gpt2
model.

Method: POST

Request Body (JSON):
    {
        "ingredients" : <string of ingredients, separated by commas>
    }

Returns (JSON):
    {
        "recipe" : <model generated recipe string>
    }
"""
@app.route('/generate-recipe-gpt2', methods=['POST'])
def generate_recipe_endpoint_gpt2():
    data = request.json
    ingredients = data.get('ingredients')

    if not ingredients:
        return jsonify({"error": "Ingredients are required"}), 400
    
    recipe = generate_recipe_gpt2(ingredients, gpt2_model, gpt2_tokenizer)

    return jsonify({"recipe": recipe})

"""
Endpoint to generate labels for up to 5 input image files, using the pretrained
object detectio model.

Method: POST

Request Body (form-data):
    Up to 5 key/value pairs such that:
        key: 'image'
        value: compressed image in JPEG format

Returns (JSON):
    {
        "ingredients": [
            {
                "label": <string label of ingredient>
            }
        ]
    }
"""
@app.route('/image-to-ingredient', methods=['POST'])
def image_to_ingredient():
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No images were provided."}), 400
        
        files = request.files.getlist('image')
        if not files or len(files) == 0:
            return jsonify({"error": "No images were provided."}), 400

        if len(files) > 5:
            return jsonify({"error": "Maximum of 5 images allowed. You have exceeded this limit."}), 400

        results = []

        for file in files:
            try:
                image = Image.open(file.stream).convert('RGB')
                predicted_label = generate_image_label(image)
                results.append({
                    "label": predicted_label
                })
            except Exception as e:
                results.append({
                    "error": str(e)
                })

        return jsonify({"ingredients": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
Endpoint to generate recipe from 5 input image files, using the pretrained
t5 OR gpt2 model.

Method: POST

Request Body (form-data):
    Up to 5 key/value pairs such that:
        key: 'image'
        value: compressed image in JPEG format

Query parameters:
    ?m = <either gpt2 or t5>

Returns (JSON):
    {
        "recipe" : <model generated recipe string>
    }
"""   
@app.route('/image-to-recipe', methods=['POST'])
def image_to_recipe():
    model_type = request.args.get('m')
    print(model_type)
    if model_type != 't5' and model_type != 'gpt2':
        return jsonify({'error' : 'Invalid model type -- either use t5 or gpt2'})
    

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No images were provided."}), 400
        
        files = request.files.getlist('image')
        if not files or len(files) == 0:
            return jsonify({"error": "No images were provided."}), 400

        if len(files) > 5:
            return jsonify({"error": "Maximum of 5 images allowed. You have exceeded this limit."}), 400

        ingredients = ""

        for file in files:
            try:
                image = Image.open(file.stream).convert('RGB')
                predicted_label = generate_image_label(image)
                ingredients += "," + predicted_label
            except Exception as e:
                return jsonify({'error' : str(e)})

        ingredients = ingredients[1:]
        if model_type == 'gpt2':
            recipe = generate_recipe_gpt2(ingredients, gpt2_model, gpt2_tokenizer)
        elif model_type == 't5':
            recipe = generate_recipe_t5(ingredients, t5_model, t5_tokenizer)

        return jsonify({'recipe' : recipe})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
Endpoint to generate labels for up to 5 input image base64 strings, using the pretrained
object detection model.

Method: POST

Request Body (JSON):
    String array with key "images" MUST contain 5 or less b64 strings
    {
        "images" : [
            "base64stringone",
            "base64stringtwo"
        ]
    }

Returns (JSON):
    {
        "ingredients": [
            {
                "label": <string label of ingredient>
            }
        ]
    }
"""    
@app.route('/image-to-ingredient-b64', methods=['POST'])
def image_to_ingredient_b64():
    try:
        data = request.json
        if not data or 'images' not in data:
            return jsonify({"error": "No images were provided."}), 400
        
        images = data['images']  # List of base64 encoded strings
        if not images or len(images) == 0:
            return jsonify({"error": "No images were provided."}), 400

        if len(images) > 5:
            return jsonify({"error": "Maximum of 5 images allowed. You have exceeded this limit."}), 400

        results = []

        for image_b64 in images:
            try:
                image_data = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_data)).convert('RGB')
                predicted_label = generate_image_label(image)
                results.append({
                    "label": predicted_label
                })
            except Exception as e:
                results.append({
                    "error": str(e)
                })

        return jsonify({"ingredients": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
Endpoint to generate recipe from 5 input image files, using the pretrained
t5 OR gpt2 model.

Method: POST

Request Body (JSON):
    String array with key "images" MUST contain 5 or less b64 strings
    {
        "images" : [
            "base64stringone",
            "base64stringtwo"
        ]
    }

Query parameters:
    ?m = <either gpt2 or t5>

Returns (JSON):
    {
        "recipe" : <model generated recipe string>
    }
"""   
@app.route('/image-to-recipe-b64', methods=['POST'])
def image_to_recipe_b64():
    model_type = request.args.get('m')
    print(model_type)
    if model_type != 't5' and model_type != 'gpt2':
        return jsonify({'error': 'Invalid model type -- either use t5 or gpt2'}), 400

    try:
        data = request.json
        if not data or 'images' not in data:
            return jsonify({"error": "No images were provided."}), 400

        images = data['images']
        if not images or len(images) == 0:
            return jsonify({"error": "No images were provided."}), 400

        if len(images) > 5:
            return jsonify({"error": "Maximum of 5 images allowed. You have exceeded this limit."}), 400

        ingredients = ""

        for image_b64 in images:
            try:
                image_data = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_data)).convert('RGB')
                predicted_label = generate_image_label(image)
                ingredients += "," + predicted_label
            except Exception as e:
                return jsonify({'error': f"Error processing one of the images: {str(e)}"}), 500

        ingredients = ingredients[1:]

        if model_type == 'gpt2':
            recipe = generate_recipe_gpt2(ingredients, gpt2_model, gpt2_tokenizer)
        elif model_type == 't5':
            recipe = generate_recipe_t5(ingredients, t5_model, t5_tokenizer)

        return jsonify({'recipe': recipe})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
