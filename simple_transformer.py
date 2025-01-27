import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformer(nn.Module):
    """
    A simple Transformer-based model for sequence processing tasks. 
    This model consists of embedding layers, positional encoding, 
    Transformer encoder layers, and a fully connected output layer.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_len):
        """
        Initializes the SimpleTransformer model.

        Parameters:
            input_dim (int): Size of the input vocabulary, representing the number 
                of unique tokens the model can handle.
            model_dim (int): Dimensionality of the token embeddings and hidden states 
                inside the Transformer.
            num_heads (int): Number of attention heads in the multi-head self-attention 
                mechanism within each Transformer encoder layer.
            num_layers (int): Number of Transformer encoder layers to stack. 
                Each layer contains a multi-head self-attention mechanism and feedforward network.
            output_dim (int): Size of the output, typically the size of the output vocabulary 
                or the number of classes in classification tasks.
            max_seq_len (int): Maximum length of the input sequences. This determines 
                the size of the positional encoding.
        """
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, model_dim))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 2)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(model_dim, output_dim)
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Defines the forward pass of the SimpleTransformer model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) 
                where each element represents a token index from the vocabulary.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim), 
                where each token in the input sequence has a corresponding output.
        """
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x)

        x = self.fc_out(x)  

        return x 

