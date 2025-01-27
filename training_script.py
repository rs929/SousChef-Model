import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from simple_transformer import SimpleTransformer
from tqdm import tqdm
import ast


def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)

def clean_string(str):
    remove_bracket = str[1:-1]
    lst = remove_bracket.split(", ")
    return lst

def convert_tensor(str):
    lst = clean_string(str)
    result = []
    for x in lst:
        result.append(clean_string(x))
    return result

def create_word_to_index_mapping(data_frame, min_frequency=1):
    word_count = {}

    for ingredients in data_frame['Ingr_Tnsr']:
        ingr_lst = clean_string(ingredients)
        for word in ingr_lst:
            word_count[int(word)] = word_count.get(word, 0) + 1

    for instructions in data_frame['Instr_Tnsr']:
        inst_lst = clean_string(instructions)
        for word in inst_lst:
            word_count[int(word)] = word_count.get(word, 0) + 1

    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_count.items():
        if count >= min_frequency:
            word_to_index[word] = len(word_to_index)

    return word_to_index

def train_model(data_frame, indexed_words, num_epochs=10, max_seq_len=128, batch_size=32):
    # Convert tokenized strings to tensors
    data_frame['Ingr_Tnsr'] = data_frame['Ingr_Tnsr'].apply(lambda x: torch.tensor([int(i) for i in clean_string(x)], dtype=torch.long))
    data_frame['Instr_Tnsr'] = data_frame['Instr_Tnsr'].apply(lambda x: torch.tensor([int(i) for i in clean_string(x)], dtype=torch.long))

    # Pad the tensors to max_seq_len
    ingr_tensor = pad_sequence(list(data_frame['Ingr_Tnsr']), batch_first=True, padding_value=indexed_words['<PAD>'])[:, :max_seq_len]
    instr_tensor = pad_sequence(list(data_frame['Instr_Tnsr']), batch_first=True, padding_value=indexed_words['<PAD>'])[:, :max_seq_len]

    # Create DataLoader for batching
    dataset = TensorDataset(ingr_tensor, instr_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleTransformer(
        input_dim=len(indexed_words),  # Ensure this matches your word_to_index length
        model_dim=256,
        num_heads=8,
        num_layers=6,
        output_dim=len(indexed_words),
        max_seq_len=max_seq_len
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (ingr_batch, instr_batch) in progress_bar:
            optimizer.zero_grad()

            # Check for maximum index in the input batch
            print("Max index in ingr_batch:", ingr_batch.max().item())

            # Ensure that we don't exceed input_dim
            if ingr_batch.max() >= len(indexed_words):
                print("Warning: Input batch contains index out of range!")
                continue  # Skip this batch if there's an issue

            # model output
            outputs = model(ingr_batch)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, len(indexed_words))
            targets = instr_batch.view(-1)

            # Loss and backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f'Epoch {epoch + 1}, Average Loss: {epoch_loss / len(data_loader)}')

    return model


if __name__ == "__main__":
    preprocessed_file_path = 'preprocessed_recipes.csv'  # Adjust path
    data_frame = load_preprocessed_data(preprocessed_file_path)


    indexed_words = create_word_to_index_mapping(data_frame, min_frequency=2)
    # print("Word to index mapping:", indexed_words)

    trained_model = train_model(data_frame, indexed_words, batch_size=32)
    print("Training complete!")