from torch.utils.data import Dataset
import numpy as np
import os
import torch

from settings import max_seq_size, dataset_name
from tokenizerLuka import Tokenizer
import pickle

#    Copyright 2025 Kenneth Wilber (kawgit)
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

class TransformerDataset(Dataset):

    def __init__(self, text_tokened):

        self.text_tokened = text_tokened

    def __len__(self):

        return len(self.text_tokened) - max_seq_size

    def __getitem__(self, idx):
        return self.text_tokened[idx:idx+max_seq_size], self.text_tokened[idx+1:idx+max_seq_size+1]
    

def load_dataset():
    encoded_file = f"encoded_{dataset_name}.pt"  # Save tensors in .pt format (PyTorch's format)
    if os.path.exists(encoded_file):
        text_indexes = torch.load(encoded_file)
    else:
        with open(dataset_name, "r") as file:
            text = file.read()
        with open("tokenizer.pkl", 'rb') as tokener:
            tokenizer = pickle.load(tokener)
        text_indexes = torch.tensor(tokenizer.encode(text))
        torch.save(text_indexes, encoded_file)
    return TransformerDataset(text_indexes)