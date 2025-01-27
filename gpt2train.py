import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

gpt_train = pd.read_csv("gpt_train.csv")
gpt_train['To_Input'].to_csv("recipes.txt", index=False, header=False)

dataset = load_dataset("text", data_files={"train": "recipes.txt"})

train_texts, val_texts = train_test_split(dataset["train"]["text"], test_size=0.1, random_state=42)

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})


tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens = {
    "pad_token": "<PAD>",
    "bos_token": "<START>",
    "eos_token": "<END>",
    "sep_token": "<SEP>"  
}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.to(device)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy() 
    return tokenized

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./gpt2-recipe",
    overwrite_output_dir=True,
    num_train_epochs=3, 
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=4,  
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500, 
    report_to="none",  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,  
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./gpt2-recipe")
tokenizer.save_pretrained("./gpt2-recipe")

model = AutoModelForCausalLM.from_pretrained("./gpt2-recipe")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-recipe")
model.to(device)

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1
)

def generate_recipe_from_user_input():
    """
    Generates a recipe based on user-provided ingredients using a pre-trained language model.

    The function prompts the user to input a list of ingredients, processes the input,
    and generates a recipe in paragraph form. The recipe is created using a language model 
    and displayed to the user.
    """

    print("Enter a list of ingredients separated by commas (e.g., chicken, salt, olive oil):")
    user_ingredients = input("Ingredients: ").strip()
    
    input_text = f"Given these ingredients: {user_ingredients}. Write a simple, clear, and complete recipe:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids,
        max_length=200,  
        temperature=0.7,  
        top_k=30,  
        top_p=0.8,  
        repetition_penalty=1.2  
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated Recipe (as a paragraph):")
    print(generated_text)

generate_recipe_from_user_input()