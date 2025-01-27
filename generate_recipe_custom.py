import torch
import torch.nn.functional as F
from simple_transformer import SimpleTransformer
from word_dict import indexed_words
import numpy as np
from generate_recipe import generate_text_from_indices, generate_recipe, adjust_model_vocab_size
import argparse

def model_setup():
  '''
  Setting up the simple transformer according to the parameters in
  generate_recipe.py, using the saved model in recipe_transformer_1.pth.
  '''
  input_dim = len(indexed_words)
  model_dim = 512
  num_heads = 8
  num_layers = 6
  output_dim = input_dim
  max_seq_len = 300

  model = SimpleTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_len)
  # model.load_state_dict(torch.load('recipe_transformer_1.pth'))
  model = adjust_model_vocab_size(model, 'recipe_transformer_1.pth', 59035)
  model.eval()
  return model

def ingr_to_idx(ingrs):
    '''
    Given an ingredient (string), maps it to and returns its index position
    in the vocabulary dictionary.
    '''
    ingr_token = indexed_words[ingrs[0]]
    return torch.tensor([ingr_token], dtype=torch.long)

if __name__ == '__main__':
    #print('in main')

    # Setting up arguments for ONE ingredient and max_recipe length
    parser = argparse.ArgumentParser(description='Input ingredient to generate_recipe.py')
    parser.add_argument('--ingr', type=str, nargs='+', help='ONE ingredient', required=True)
    parser.add_argument('--max_length', type=int, default=50, help='recipe max_length')
    
    args = parser.parse_args()

    model = model_setup()

    arg_ingr = args.ingr
    input_tensor = ingr_to_idx(arg_ingr)

    #printing output
    output = generate_recipe(model, input_tensor, args.max_length)
    output_text= generate_text_from_indices(output.tolist())

    print(f"Output Recipe:\n{output_text}")