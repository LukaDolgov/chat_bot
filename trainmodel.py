import numpy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
import re
from transformermine import Transformer
from transformermine import get_device

#old file for training the model

# d_model = 512 #set vector length
# num_heads = 8 #amount of attention heads
# drop_prob = 0.1 #rate of dropout
#batch_size #amount of examples (sentences) per run through of the whole net and backpropogation process
# max_sequence_length = 200 #max words to send through encoder, if using less the rest go in as "padding"
# ffn_hidden = 1000 #amount of neurons in feed forward linear layer
# num_layers = 5 #number of repeated transformer encoder units in architecture


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



def split_text2(text):
    # Define special tokens
    special_tokens = ["<pad>", "<start>", "<end>"]

    # Initialize the result list to store tokenized words
    result = []
    
    # Iterate through the text and handle special tokens first
    for token in special_tokens:
        if token in text:
            # Split around special tokens and append them separately
            parts = text.split(token)
            for part in parts[:-1]:
                # Handle regular text in parts
                result.extend(re.findall(r'\w+|[^\w\s]', part))  # Word characters and punctuation
            result.append(token)  # Add the special token itself
            text = parts[-1]  # Remainder of the text after the last special token
    
    # Use regular expression to split the remaining text into words and punctuation
    # This regex will match words (\w+) or punctuation ([^\w\s]) or spaces
    tokens = re.findall(r'\w+|[^\w\s]', text)
    
    # Add remaining tokens to the result
    result.extend(tokens)
    
    return result

english_file = "shakespeare.txt"
with open(english_file, 'r') as file:
    english_sentences = file.readlines()
    file.seek(0)
    text = file.read().lower()
english_vocabulary = list(set(split_text2(text)))
#print(english_vocabulary)
TOTAL_SENTENCES = 40000
english_sentences = [sentence.rstrip('\n').lower() for sentence in english_sentences]
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}
vocab_size = len(index_to_english)
english_sentences = english_sentences[:TOTAL_SENTENCES]
#print(english_sentences[:5])

#params
d_model = 512
batch_size = 128
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 8
max_sequence_length = 40 
kn_vocab_size = len(english_vocabulary)
PADDING_TOKEN = "<pad>"


def is_valid_length(sentence, max_sequence_length):
    return len(list(set(split_text2(sentence)))) < (max_sequence_length) and len(sentence) > 1

valid_sentence_indicies = []
for index in range(len(english_sentences)):
    english_sentence = english_sentences[index]
    if is_valid_length(english_sentence, max_sequence_length) and len(list(set(split_text2(english_sentences[index])))) > 1:
        valid_sentence_indicies.append(index)
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

print(english_sentences[:3])

from torch.utils.data import Dataset, DataLoader
import torch


class TextDataset(Dataset):

    def __init__(self, english_sentences, max_token_length):
        self.english_sentences = english_sentences
        self.max_token_length = max_token_length
        self.padding_token = PADDING_TOKEN
        self.inps = []
        self.oups = []
        self.get_inps(max_token_length)
        
    def get_inps(self, max_seq_length):
        for sentence in self.english_sentences:
            tokens = list(split_text2(sentence))  # Assuming split_text2(tokenizes sentence)
            
            # Handle padding for sentences shorter than max_seq_length
            input_tokens = tokens[:len(tokens) - 1]  # Take only up to max_seq_length tokens
            output_tokens = tokens[1:]  # Shift tokens for output
            input_padded = input_tokens + [self.padding_token] * (max_seq_length - len(input_tokens))
            output_padded = output_tokens + [self.padding_token] * (max_seq_length - len(output_tokens))
            self.inps.append(input_padded)
            self.oups.append(output_padded)
# part of one batch is in form [inps, inps - 1 shifted]
        
    
    def __len__(self):
        return len(self.inps)

    def __getitem__(self, idx):
        return self.inps[idx], self.oups[idx]
dataset = TextDataset(english_sentences, max_sequence_length)
    
print(dataset[1])
batch_size = 128 
train_loader = DataLoader(dataset, batch_size, shuffle=True)
iterator = iter(train_loader)

batch = []
for batch_num, batch in enumerate(iterator):
    print(batch)
    batch = batch
    if batch_num > 3:
        break


def batch_tokenize(batch):
    def tokenize(sentence, language_to_index):
        # Tokenize the sentence into words
        sentence_words = list(split_text2(sentence))  # split by spaces to get words
        # Start token logic
        sentence_word_indicies = [] 
        # Add the words from the sentence
        sentence_word_indicies.extend([language_to_index[token] for token in sentence_words])
        while len(sentence_word_indicies) < max_sequence_length:
            sentence_word_indicies.append(language_to_index["<pad>"])  # Add padding token
        if len(sentence_word_indicies) > max_sequence_length:
            sentence_word_indicies = sentence_word_indicies[:max_sequence_length]
        return torch.tensor(sentence_word_indicies)
    inp_tokenized = []
    oup_tokenized = []
    for sentence_num in range(batch_size):
        eng_sentence = batch[0][sentence_num]
        eng2_sentence = batch[1][sentence_num]
        inp_tokenized.append( tokenize(eng_sentence, english_to_index))
        oup_tokenized.append( tokenize(eng2_sentence, english_to_index))
    inp_tokenized = torch.stack(inp_tokenized)
    oup_tokenized = torch.stack(oup_tokenized)
   # print(inp_tokenized)
   # print(oup_tokenized)

NEG_INFTY = -1e9

def create_masks(x_batch):
    num_sentences = len(x_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    for idx in range(num_sentences):
      eng_sentence_length = len(x_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    combined_mask = torch.logical_or(look_ahead_mask, encoder_padding_mask)
    
    # Apply the mask: where True (in combined_mask), set NEG_INFTY to block attention
    encoder_self_attention_mask = torch.where(combined_mask, NEG_INFTY, 0)
    return encoder_self_attention_mask

transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          kn_vocab_size,
                          english_to_index,
                          PADDING_TOKEN)


criterian = nn.CrossEntropyLoss(ignore_index=english_to_index[PADDING_TOKEN],
                                reduction='none')
# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_bot():
    transformer.train()
    transformer.to(device)
    total_loss = 0
    num_epochs = 10  # Change epochs here
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        iterator = iter(train_loader)
        for batch_num, batch in enumerate(iterator):
            # Get the inputs and targets from the batch
            x_batch = batch[0]
            y_batch = batch[1]
            optim.zero_grad()
            # Initialize the total loss for this batch
            batch_loss = 0
            # Loop through each token in the batch
            for j in range(1, len(x_batch[0])):
                for i in range(1, len(x_batch[0][j])):  # Start from the second token to predict
                    # Slice the batch for the first i tokens
                    x_input = x_batch[0][j][:i]  # Take first i tokens as input
                    y_target = y_batch[0][j][i+1]  # The i-th token is the target
                    # Create self-attention mask for the input
                    encoder_self_attention_mask = create_masks(x_input)
                    # Get model predictions for the input sequence
                    predicted_word = transformer(x_input, encoder_self_attention_mask)
                    labels = transformer.encoder.sentence_embedding.batch_tokenize(y_target)
                    loss = criterian(
                    predicted_word.view(-1, kn_vocab_size).to(device),
                    labels.view(-1).to(device)
                        ).to(device)
                    valid_indicies = torch.where(labels.view(-1) == english_to_index[PADDING_TOKEN], False, True)
                    loss = loss.sum() / valid_indicies.sum()
                    batch_loss += loss
            # Average the loss over the batch
            batch_loss = batch_loss / len(x_batch)
            # Backpropagation
            batch_loss.backward()
            optim.step()
            total_loss += batch_loss.item()
            # Logging and saving the model periodically
            if batch_num % 100 == 0:
                print(f"Saving model at iteration {batch_num}, epoch {epoch}")
                torch.save(transformer.state_dict(), 'model_weights.pth')
            if batch_num % 20 == 0:  # Can change this value to see more progress
                print(f"Iteration {batch_num}: Loss = {batch_loss.item()}")
                print(f"Input: {x_batch[0]}")
                print(f"Real next token: {y_batch[0]}")
    # Save the final model after training
    torch.save(transformer.state_dict(), 'model_weights.pth')
train_bot()


