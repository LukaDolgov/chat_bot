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


input_text = """Sir Wilber pranced along all day. He wondered what Luka was doing! "Oh I'm so glad to be happy on this fine day!"
 - Sir Wilber thought. As he thought and he wondered and was so very glee, he wandered on over to the strange tree. 
 """
 
input_text2 = """
Mon cour was so beautiful, Luka didn't know what to do. 
He could only spend his days loving true.
For her eyes were like diamonds in the coldest night.
And her warm emebrace made it all alright.
To kiss her gently as time goes by.
Is a dream I will carry on, I must try.
"""

with open("tokenizer.pkl", 'rb') as tokener:
    tokenizer = pickle.load(tokener)
text_tokens = tokenizer.encode(input_text2)


if (len(text_tokens) >= max_seq_size): 
    print("all good")
    input_seq_size = len(text_tokens)
    transformer = Transformer(
                    embedding_size, 
                    ffn_hidden, 
                    num_heads, 
                    drop_prob, 
                    num_layers,
                    max_seq_size, 
                    vocab_size,)

    transformer.load_state_dict(torch.load('model_weights.pth'))

    #print(len(text_tokens))
    new_encoded_text = transformer.generate2(text_tokens, amount_to_generate, temperature, top_k)

    #print(tokenizer.itob)
    print("Output: \n")
    print(tokenizer.decode(text_tokens))
else:
    print("please add more tokens to input")

