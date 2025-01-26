import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

#old file for transformer
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def is_word_char(c):
    return c.isalnum() or c == '_' or c == "'"

# Function to identify if a character is punctuation
def is_punctuation(c):
    return c in ['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '-', '...', '@', '#', '$', '%', '&', '/', '\\']

# Function to split text into words and punctuation while keeping special tokens intact
def split_text(text):
    result = []
    word = ''
    special_token = False  # Flag to track if we're processing a special token
    
    # Check if any of the special tokens are in the text, and treat them as a single token
    special_tokens = ["<pad>", "<start>", "<end>"]
    
    i = 0
    while i < len(text):
        # Look for special tokens first
        for token in special_tokens:
            if text[i:i+len(token)] == token:
                result.append(token)  # Append special token as a single unit
                i += len(token)  # Skip over the special token
                special_token = True
                break
        
        if not special_token:
            char = text[i]
            if is_word_char(char):
                word += char  # Accumulate the word
            elif is_punctuation(char):
                if word:  # If there's a current word being formed, append it
                    result.append(word)
                    word = ''
                result.append(char)  # Append the punctuation
            else:
                if word:  # If a word was being formed, append it before breaking
                    result.append(word)
                    word = ''
            special_token = False  # Reset the flag if not a special token

        i += 1
    
    if word:  # Add any word left at the end
        result.append(word)

    return result




def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch):
        def tokenize(tokens, language_to_index):
            # Start token logic
            sentence_word_indicies = []
            # Add the words from the sentence
            sentence_word_indicies.extend([language_to_index[token] for token in tokens])
            while len(sentence_word_indicies) < self.max_sequence_length:
                sentence_word_indicies.append(language_to_index["<pad>"])  # Add padding token
            if len(sentence_word_indicies) > self.max_sequence_length:
                sentence_word_indicies = sentence_word_indicies[:self.max_sequence_length]
            return torch.tensor(sentence_word_indicies)
        tokenized = []
        for sentence_num in range(len(batch)):
            batch_part = batch[sentence_num]
            tokenized.append( tokenize(batch_part, self.language_to_index))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x): # sentence
        x = self.batch_tokenize(x)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
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
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask):
        x = self.sentence_embedding(x)
        x = self.layers(x, self_attention_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                kn_vocab_size,
                english_to_index,
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, inpx, mask):
        x = self.encoder(inpx, mask)
        out = self.linear(x)
        return out