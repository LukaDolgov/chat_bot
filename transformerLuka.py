import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
import random
from settings import *
from tokenizerLuka import Tokenizer

NEG_INF = -1e9

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(scaled.size(0), scaled.size(1), -1, -1)
        scaled = scaled + mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, input_seq_size):
        super().__init__()
        self.input_seq_size = input_seq_size
        self.embedding_size = embedding_size

    def forward(self):
        even_i = torch.arange(0, self.embedding_size, 2).float()
        denominator = torch.pow(10000, even_i/self.embedding_size)
        position = (torch.arange(self.input_seq_size)
                          .reshape(self.input_seq_size, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class TokenEncoder(nn.Module):
    def __init__(self, input_seq_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.position_encoder = PositionalEncoding(embedding_size, input_seq_size)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, tokens): # sentence
        x = self.embedding(tokens)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_seq_size, embedding_size, num_heads):
        super().__init__()
        self.input_seq_size = input_seq_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.qkv_layer = nn.Linear(embedding_size , 3 * embedding_size)
        self.linear_layer = nn.Linear(embedding_size, embedding_size)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(input_seq_size, input_seq_size)).float() * NEG_INF  # Large negative value
        )
    
    def forward(self, x):
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, self.input_seq_size, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        causal_mask = self.causal_mask[:self.input_seq_size, :self.input_seq_size]
        values, attention = scaled_dot_product(q, k, v, causal_mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, self.input_seq_size, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_size, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden)
        self.linear2 = nn.Linear(hidden, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, input_seq_size, embedding_size, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(input_seq_size, embedding_size=embedding_size, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[embedding_size])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(embedding_size=embedding_size, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[embedding_size])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x.clone()
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Module):
    def __init__(self, *modules):
        super(SequentialEncoder, self).__init__()
        self.model = nn.ModuleList(modules)  # Use ModuleList to hold the layers

    def forward(self, x):
        for module in self.model:
            x = module(x)
            # Pass the input through each layer sequentially
        return x

class TransformerLayer(nn.Module):
    def __init__(self, 
                 embedding_size, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 inp_seq_size):
        super().__init__()
        self.sentence_embedding = TokenEncoder(inp_seq_size, embedding_size)
        self.layers = SequentialEncoder(*[EncoderLayer(inp_seq_size, embedding_size, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x):
        x = self.sentence_embedding(x)
        x = self.layers(x)
        return x


class Transformer(nn.Module):
    def __init__(self, 
                embedding_size, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                input_seq_size, 
                vocab_size,
                ):
        super().__init__()
        self.transformerlayer = TransformerLayer(embedding_size, ffn_hidden, num_heads, drop_prob, num_layers, input_seq_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, inpx):
        x = self.transformerlayer(inpx) #shape (batch_size, seq_length, embedding)
        out = self.linear(x) #shape (batch, seq_length, vocab_size)
        print("final output vector size: " + f"{out.size()}")
        return out
    
    def generate(self, start_text, num_new, temperature, top_k):
        self.eval()
        text_tokens = start_text
        for i in range(num_new):
            input_tokens = text_tokens
            # input_tokens = pad_sequence(input_tokens, max_seq_size, padding_token_index)
            input_tokens_tensor = torch.tensor(input_tokens[-max_seq_size:]).reshape(1, -1)
            input_tokens_tensor = input_tokens_tensor.repeat(16, 1)
            #print(input_tokens_tensor.size())  # Debugging the size

            with torch.no_grad():
                logits = self.forward(input_tokens_tensor)  # Forward pass
            
            # Extract the logits for the last token in the sequence
            output_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            # Apply temperature scaling if needed
            
            # Apply softmax to get probabilities
            output_probs = torch.softmax(output_logits, dim=-1)
            output_probs = output_probs[0] 
            #print(output_probs)
            top_k_values, top_k_indices = torch.topk(output_probs, top_k)
            print(top_k_indices)
            index_submit = top_k_indices[0].item()
            if random.random() > temperature:
                random_index = random.randint(0, top_k_indices.size()[0] - 1)
                index_submit = top_k_indices[random_index].item()  # Get the actual token index.
            # Append the new token to the sequence
            text_tokens.append(index_submit)
            print(text_tokens)

        return text_tokens


def pad_sequence(tokens, max_seq_size, pad_token=5):
    if len(tokens) < max_seq_size:
        padding_needed = max_seq_size - len(tokens)
        tokens = tokens + [pad_token] * padding_needed
    tokens = tokens[-max_seq_size:]
    return tokens

#need to add masking to prevent padding from affecting softmax, consider removing it from output list before caluclation,
#figure out which index is assigned to pad, or dedicate one

#consider strategy of feeding "hidden tokens" before input in order to input any length.