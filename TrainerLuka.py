import time
import wandb
import torch
from settings import dataset_name, max_seq_size, vocab_size

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NEG_INFINITY = 1e-9



class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.batch_xs = None
        self.batch_ys = None
        self.batch_outputs = None
        self.batch_index = None
        self.batch_loss = None
        self.epoch_index = None
        self.epoch_loss = None 
        self.attention_mask = None
        self.logits = None
        self.y_check = None

        self.batch_losses = []
        self.epoch_losses = []
        
        
        
      
    def fit(self, num_epochs):

        
        self.model.train()
        self.model.to(device)
        for epoch in range(num_epochs):
            epoch_loss_total = 0
            self.batch_losses = []
            for self.batch_idx, (self.batch_xs, self.batch_ys) in enumerate(self.dataloader):
                self.batch_xs = self.batch_xs.to(device)
               # print("batch size: " + f"{self.batch_xs.size()}")
                self.batch_ys = self.batch_ys.to(device)
                self.optimizer.zero_grad()
                print(self.batch_xs.size())
                self.batch_outputs = self.model(self.batch_xs)
                
                self.logits = self.batch_outputs.reshape(-1, vocab_size)  # Flatten to (batch_size * seq_length, vocab_size)
                self.y_check = self.batch_ys.reshape(-1)  # Flatten to (batch_size * seq_length,)
               # print("logits size: " + f"{self.logits.size()}")
                #print("y_check size: " + f"{self.y_check.size()}")
                
                loss = self.criterion(self.logits, self.y_check)
                loss.backward()
                self.optimizer.step()
                
                
                self.batch_loss = loss.item()
                self.batch_losses.append(self.batch_loss)

                epoch_loss_total += self.batch_loss
                self.epoch_loss = epoch_loss_total / (self.batch_idx + 1)

                self.epoch_losses.append(self.epoch_loss)
                if self.batch_idx % 10 == 0:
                    print(f"Saving model at iteration {self.batch_idx}, epoch {epoch}")
                    torch.save(self.model.state_dict(), 'model_weights.pth')
                if self.batch_idx % 10 == 0:  # Can change this value to see more progress
                    print(f"Iteration {self.batch_idx}: Loss = {self.batch_loss}")
                    print(f"Input: {self.batch_xs[0]}")
                    print(f"Real next token: {self.batch_ys[0]}")
        print("finished training!")
        torch.save(self.model.state_dict(), 'model_weights.pth')


            
        
