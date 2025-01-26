from transformermine import Transformer
from transformermine import get_device
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import random
#old file for using the transformer
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
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}
vocab_size = len(index_to_english)


d_model = 512
batch_size = 128
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 8
max_sequence_length = 40
kn_vocab_size = len(english_vocabulary)
PADDING_TOKEN = "<pad>"



NEG_INFTY = -1e9

def create_masks_no_look(x_batch):
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
    return encoder_self_attention_mask

def sample_with_temperature(logits, temperature=1.0):
    """
    Sample the next token from the logits using temperature sampling.
    
    Args:
        logits: The raw output logits from the model (before softmax).
        temperature: The temperature for sampling. A higher temperature increases randomness.
    
    Returns:
        The index of the sampled token.
    """
    # Scale logits by temperature
    logits = logits / temperature
    
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample a token from the probability distribution
    token = torch.multinomial(probs, 1)  # Sample one token
    return token.item()  # Return the index of the token

def generate_text(model, start_text, max_length=50, temperature=1.0):
    model.eval()  # Set the model to evaluation mode
    
    # Tokenize the start text (you should already have a function for this)
    input_tokens = list(split_text2(start_text))
    
    generated_text = start_text
    encoder_self_attention_mask = create_masks_no_look(input_tokens)
    
    for i in range(max_length):
        # Get the predicted word probabilities
        print(len(input_tokens))
        random_input_tokens = input_tokens[:i+1] 
        encoder_self_attention_mask = create_masks_no_look(random_input_tokens)
        with torch.no_grad():
            output = model(random_input_tokens, encoder_self_attention_mask)  # Get the model output
            last_token_logits = output[0, -1, :]  # Get the last token's logits
        
        # Sample the next word using temperature
        next_word_index = sample_with_temperature(last_token_logits, temperature=temperature)
        
        # Convert the predicted index back to a word
        predicted_word = index_to_english[next_word_index]
        
        if predicted_word == '.' or predicted_word == "," or predicted_word == "?" or predicted_word == ":":
            generated_text += predicted_word
        else:
            generated_text += ' ' + predicted_word
        # Append the predicted word to the generated text
        
        # Update input_tokens for the next iteration (adding the predicted word)
        input_tokens = list(split_text2(generated_text))
        encoder_self_attention_mask = create_masks_no_look(input_tokens)
    
    return generated_text



def main():
    transformer = Transformer(d_model, 
                            ffn_hidden,
                            num_heads, 
                            drop_prob, 
                            num_layers, 
                            max_sequence_length,
                            kn_vocab_size,
                            english_to_index,
                            PADDING_TOKEN)
    transformer.load_state_dict(torch.load('model_weights.pth'))
    # Example usage:
    start_text = "What should I do?".lower()
    generated_text = generate_text(transformer, start_text, max_length=50, temperature=.8)
    print(generated_text)
main()