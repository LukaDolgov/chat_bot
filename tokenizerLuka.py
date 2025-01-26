import random
from collections import Counter
from settings import vocab_size, dataset_name
from tqdm import tqdm
import pickle

path_write = "tokenizer.pkl"

class Tokenizer():
    
    def __init__(self):
        self.vocab_size = vocab_size  # Maximum size of the vocabulary
        self.btoi = {i: i for i in range(256)}  # Initialize byte-to-index mapping for characters
        self.itob = {i: bytes([i]) for i in range(256)}  # Reverse mapping (index to byte)
    
    def train(self, text_to_train):
        text_bytes = bytes(text_to_train, 'utf-8')
        text_indexes = [self.btoi[i] for i in text_bytes]
       # print(self.btoi)
     #   print(self.itob)
        while len(self.btoi) < self.vocab_size: #repeat until vocab size met
            with tqdm(total=self.vocab_size - len(self.btoi)) as pbar:
                print(len(self.btoi))
                print(len(self.itob))
                pair_counts = Counter(zip(text_indexes[:-1], text_indexes[1:])) #find pairs
                common_pair = pair_counts.most_common(1)[0][0]
                print(common_pair)
                #merge most common pair
                best_pair_bytes = self.itob[common_pair[0]] + self.itob[common_pair[1]]
                new_index = len(self.btoi)
                self.btoi[best_pair_bytes] = new_index 
                self.itob[new_index] = best_pair_bytes
                #replace pair in text
                i = 0
                updated_text_indexes = []
                while i < len(text_indexes) - 1:
                    if text_indexes[i] == common_pair[0] and text_indexes[i+1] == common_pair[1]:
                        updated_text_indexes.append(new_index)
                        i += 2  
                    else:
                        updated_text_indexes.append(text_indexes[i])
                        i += 1
                    if i == len(text_indexes) - 1:
                        updated_text_indexes.append(text_indexes[i])
                text_indexes = updated_text_indexes
                pbar.update(1)  # Update the progress bar after each iteration
        print(self.itob)
        print(len(self.btoi) == len(self.itob))
        if b"F" in self.itob.values():
            print("b'F' is in itob")
        else:
            print("b'a' is not in itob")
    def encode(self, text, dropout=.1):
        text_bytes = bytes(text, 'utf-8')  # Convert the input text into bytes
        text_indexes = []
        i = 0
        
        with tqdm(total=len(text_bytes)) as pbar:
            while i < len(text_bytes):
                matched = False
                for length in range(len(text_bytes) - i, 0, -1): 
                    token = text_bytes[i:i+length]
                    if token in self.btoi:
                        if random.random() > dropout:
                            text_indexes.append(self.btoi[token]) 
                        i += length 
                        matched = True
                        break
                if not matched:
                    i += 1  
                pbar.update(1)  
            return text_indexes

    def decode(self, text_indexes):
        text_bytes = b''.join([self.itob[index] for index in text_indexes])  
        return text_bytes.decode('utf-8')  


            

def main_train():
    with open (dataset_name, "r") as file:
        text_to_train = file.read()
    tokenizer = Tokenizer()
    tokenizer.train(text_to_train)
    with open("tokenizer.pkl", 'wb') as file:
        pickle.dump(tokenizer, file)
main_train()