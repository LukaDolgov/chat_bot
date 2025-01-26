from torch.utils.data import DataLoader
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from dataset import load_dataset
from settings import *
from TrainerLuka import Trainer
from transformerLuka import Transformer



dataset = load_dataset()
transformer = Transformer(
                embedding_size, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_seq_size, 
                vocab_size,)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(transformer.parameters(), lr=learning_rate)

trainer = Trainer(
    transformer,
    DataLoader(dataset, batch_size, shuffle=True, drop_last=True),
    criterion,
    optim
)

trainer.fit(epochs)


