import torch
import torch.nn.functional as F
from simple_transformer import SimpleTransformer
from word_dict import indexed_words, reverse_vocab
import numpy as np

index_to_word = reverse_vocab
word_to_index = indexed_words

pad_token_idx = word_to_index['<PAD>']

def generate_text_from_indices(indices):
    """Converts indices to text, handling repetitive or unwanted tokens."""
    words = [index_to_word[idx] for idx in indices if idx in index_to_word and index_to_word[idx] not in ('<START>', '<PAD>')]
    return ' '.join(words)

import torch
import torch.nn.functional as F

def generate_recipe(model, input_tensor, max_length, top_k=10):
    """Generates a recipe with top-k sampling for more coherent output."""
    model.eval()
    generated_tokens = input_tensor

    for step in range(max_length):
        outputs = model(generated_tokens.unsqueeze(0))  
        next_token_logits = outputs[:, -1, :]  
        
        top_k_probs, top_k_indices = torch.topk(F.softmax(next_token_logits, dim=-1), top_k)
        next_token = top_k_indices[0, torch.multinomial(top_k_probs, 1)]  

        next_token = next_token.view(1)  
       
        if next_token.item() == word_to_index['<END>']:
            break

        generated_tokens = torch.cat((generated_tokens, next_token), dim=0)

    return generated_tokens



def adjust_model_vocab_size(model, checkpoint_path, expected_vocab_size):
    """Adjusts the model's layers to match the size in the checkpoint."""
    state_dict = torch.load(checkpoint_path, weights_only=True)

    embedding_weight_size = state_dict['embedding.weight'].size(0)
    fc_out_weight_size = state_dict['fc_out.weight'].size(0)

    if model.embedding.weight.size(0) > embedding_weight_size:
        model.embedding.weight = torch.nn.Parameter(model.embedding.weight[:embedding_weight_size])
        model.fc_out.weight = torch.nn.Parameter(model.fc_out.weight[:fc_out_weight_size])
        model.fc_out.bias = torch.nn.Parameter(model.fc_out.bias[:fc_out_weight_size])
    elif model.embedding.weight.size(0) < embedding_weight_size:
        pass

    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    input_dim = len(indexed_words)
    model_dim = 512
    num_heads = 8
    num_layers = 6
    output_dim = input_dim
    max_seq_len = 50

    model = SimpleTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_len)
    # model.load_state_dict(torch.load('recipe_transformer_1.pth'))
    model = adjust_model_vocab_size(model, 'recipe_transformer_1.pth', 59035)
    model.eval()
    export_model = model

    # Use a valid start token here
    start_token = torch.tensor([indexed_words['chicken']])
    input_tensor = start_token

    # Generate the recipe
    max_length = 50
    generated_indices = generate_recipe(model, input_tensor, max_length)

    generated_recipe = generate_text_from_indices(generated_indices.tolist())

    print("Generated Recipe:")
    print(generated_recipe)
