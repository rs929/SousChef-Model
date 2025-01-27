import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from simple_transformer import SimpleTransformer
import ast
import torch.nn.functional as F
from tqdm import tqdm

class RecipeDataset(Dataset):
    """
    A dataset class for handling tokenized recipes, specifically instructions and ingredients.
    """
    def __init__(self, data, max_seq_len):
        """
        Initializes the RecipeDataset class.

        Parameters:
            data (pd.DataFrame): DataFrame containing 'Instr_Tnsr' and 'Ingr_Tnsr' columns with tokenized tensors.
            max_seq_len (int): Maximum sequence length for padding/truncating.
        """
        self.instructions = data['Instr_Tnsr'].tolist()
        self.ingredients = data['Ingr_Tnsr'].tolist()
        self.max_seq_len = max_seq_len

    def __len__(self):
        """
        Returns the number of recipes in the dataset.

        Returns:
            int: The total number of recipes.
        """
        return len(self.instructions)

    def __getitem__(self, idx):
        """
        Retrieves the instruction and ingredient tensors for the given index, with padding or truncating applied.

        Parameters:
            idx (int): Index of the recipe to retrieve.

        Returns:
            tuple: A tuple containing (instr_tensor, ingr_tensor)
        """
        instr_tensor = self.instructions[idx]
        ingr_tensor = self.ingredients[idx]

        if len(instr_tensor) > self.max_seq_len:
            instr_tensor = instr_tensor[:self.max_seq_len]
        else:
            instr_tensor = F.pad(instr_tensor, (0, self.max_seq_len - len(instr_tensor)), value=0)

        if len(ingr_tensor) > self.max_seq_len:
            ingr_tensor = ingr_tensor[:self.max_seq_len]
        else:
            ingr_tensor = F.pad(ingr_tensor, (0, self.max_seq_len - len(ingr_tensor)), value=0)

        return instr_tensor, ingr_tensor

def load_data(file_path):
    """
    Loads and processes the recipe data from a CSV file, converting tokenized columns from string to tensors.

    Parameters:
        file_path (str): Path to the CSV file containing preprocessed recipe data.

    Returns:
        pd.DataFrame: DataFrame with tokenized instruction and ingredient tensors.
    """
    data = pd.read_csv(file_path)

    def convert_to_tensor(column):
        return [torch.tensor(ast.literal_eval(item)) for item in column]

    data['Instr_Tnsr'] = convert_to_tensor(data['Instr_Tnsr'])
    data['Ingr_Tnsr'] = convert_to_tensor(data['Ingr_Tnsr'])

    return data

def get_vocabulary_size(dataset):
    """
    Computes the vocabulary size of the dataset by finding the largest token index across both instructions and ingredients.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing tokenized tensors in 'Instr_Tnsr' and 'Ingr_Tnsr' columns.

    Returns:
        int: The size of the vocabulary, which is the maximum token index plus one.
    """
    instr_tensors = torch.cat([instr.clone().detach() for instr in dataset['Instr_Tnsr']])
    ingr_tensors = torch.cat([ingr.clone().detach() for ingr in dataset['Ingr_Tnsr']])
    
    max_instr_index = instr_tensors.max().item()
    max_ingr_index = ingr_tensors.max().item()

    return max(max_instr_index, max_ingr_index) + 1

def validate_model(model, dataloader):
    """
    Performs validation on the model by computing the average loss over the validation dataset.

    Parameters:
        model (SimpleTransformer): The trained Transformer model.
        dataloader (DataLoader): DataLoader containing the validation data.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for ingr, instr in dataloader:
            outputs = model(ingr) 
            loss = criterion(outputs.view(-1, outputs.size(-1)), instr.view(-1))  
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss



def train_model(model, train_loader, val_loader, num_epochs, learning_rate, save_path):
    """
    Trains the model using the training data and validates it using the validation data. 
    Saves the model with the lowest validation loss during training.

    Parameters:
        model (SimpleTransformer): The Transformer model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        num_epochs (int): The number of epochs to train the model.
        learning_rate (float): The learning rate for the optimizer.
        save_path (str): Path to save the model with the best validation loss.

    Returns:
        None
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        
        model.train()

        for ingr, instr in tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False):
            optimizer.zero_grad()
            outputs = model(ingr)  
            loss = criterion(outputs.view(-1, outputs.size(-1)), instr.view(-1))  
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Training Loss: {avg_train_loss:.4f}')

        val_loss = validate_model(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    data = load_data('preprocessed_recipes.csv')

    input_dim = get_vocabulary_size(data)

    model_dim = 512
    num_heads = 8
    num_layers = 6
    output_dim = input_dim
    max_seq_len = 300

    recipe_dataset = RecipeDataset(data, max_seq_len)

    train_size = int(0.8 * len(recipe_dataset))
    val_size = len(recipe_dataset) - train_size
    train_dataset, val_dataset = random_split(recipe_dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_len)

    save_path = 'recipe_transformer_1.pth'
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, save_path=save_path)

    ### NOTE: Use the following code to load the model for use later:
    # model = SimpleTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_len)
    # model.load_state_dict(torch.load('best_recipe_model.pth'))
    # model.eval()