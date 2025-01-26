from torch.utils.data import Dataset
import numpy as np
import os
import torch

from settings import max_seq_size, dataset_name
from tokenizerLuka import Tokenizer
import pickle

class TransformerDataset(Dataset):

    def __init__(self, text_tokened):

        self.text_tokened = text_tokened

    def __len__(self):

        return len(self.text_tokened) - max_seq_size

    def __getitem__(self, idx):
        return self.text_tokened[idx:idx+max_seq_size], self.text_tokened[idx+1:idx+max_seq_size+1]
    

def load_dataset():

    if os.path.exists(f"datasets/{dataset_name}.npy"):
        text_indexes = torch.tensor(np.load(f"datasets/{dataset_name}.npy"))
    else:
        with open (dataset_name, "r") as file:
            text = file.read()
        with open("tokenizer.pkl", 'rb') as tokener:
            tokenizer = pickle.load(tokener)  # Load the object from the pickle file
        text_indexes = torch.tensor(tokenizer.encode(text))

    return TransformerDataset(text_indexes)