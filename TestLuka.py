from settings import dataset_name
from tokenizerLuka import Tokenizer
from transformerLuka import Transformer
import pickle
from settings import *
import torch


transformer = Transformer(
                embedding_size, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_seq_size, 
                vocab_size,)

input_text = """
Sir Wilber pranced along all day. He wondered what Luka was doing! "Oh I'm so glad to be happy on this fine day!"
 - Sir Wilber thought. As he thought and he wondered and was so very glee, he wandered on over to the strange tree. 
 """

transformer.load_state_dict(torch.load('model_weights.pth'))
with open("tokenizer.pkl", 'rb') as tokener:
    tokenizer = pickle.load(tokener)
text_tokens = tokenizer.encode(input_text)

print(len(text_tokens))
new_encoded_text = transformer.generate(text_tokens, amount_to_generate, temperature)

print(tokenizer.itob)

print("Output: \n")
print(tokenizer.decode(new_encoded_text))


