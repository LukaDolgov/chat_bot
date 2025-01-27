import random
from collections import Counter
from settings import vocab_size, dataset_name
from tqdm import tqdm
import pickle

path_write = "tokenizer.pkl"

class Tokenizer():
    
    def __init__(self):
        self.vocab_size = vocab_size  # Maximum size of the vocabulary
        self.itob = {i: bytes([i]) for i in range(256)}  # Reverse mapping (index to byte)
        self.btoi = {value: key for key, value in self.itob.items()}
    
    def train(self, text_to_train):
        #print(self.btoi)
        #print(self.itob)
        text_bytes = bytes(text_to_train, 'utf-8')
        text_indexes = [self.btoi[bytes([i])] for i in text_bytes]
        while len(self.btoi) < self.vocab_size: #repeat until vocab size met
            with tqdm(total=self.vocab_size - len(self.btoi)) as pbar:
                # print(len(self.btoi))
               # print(len(self.itob))
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
        print(self.btoi)
        print(self.itob)
        if b"S" in self.btoi:
            print("b'S' is in btoi")
        else:
            print("b'S' is not in btoi")
    def encode(self, text):
        text_bytes = bytes(text, 'utf-8')  # Convert the input text into bytes
        text_indexes = []  # List to store the encoded indices
        i = 0  # Pointer to track the current position in the byte sequence
        longest_token_length = max(len(token) for token in self.btoi.keys())
        with tqdm(total=len(text_bytes), desc="Encoding") as pbar:
            while i < len(text_bytes):
                matched = False
                max_check_length = min(longest_token_length, len(text_bytes) - i)
                for length in range(max_check_length - i, 0, -1):  # Adjust max token length as needed
                    token = text_bytes[i:i+length]
                    if token in self.btoi:  # Use btoi for token lookup
                        text_indexes.append(self.btoi[token])  # Append the corresponding index
                        i += length  # Move the pointer forward
                        matched = True
                        pbar.update(length)  # Update the progress bar
                        break  # Exit the inner loop once a match is found
                
                if not matched:
                    # Handle unknown tokens
                    if b'<UNK>' in self.btoi:
                        # If <UNK> token exists in the vocabulary, use it
                        text_indexes.append(self.btoi[b'<UNK>'])
                    else:
                        # Fall back to encoding individual bytes
                        single_byte = text_bytes[i:i+1]
                        if single_byte in self.btoi:
                            text_indexes.append(self.btoi[single_byte])
                        else:
                            # If even the single byte is not in the vocabulary, raise an error
                            raise ValueError(f"Byte not found in vocabulary: {single_byte}")
                    i += 1  # Move the pointer forward by 1 byte
                    pbar.update(1)  # Update the progress bar
        
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