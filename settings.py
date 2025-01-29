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

#datasets
dataset_name = "shakespeare.txt"
saved_input = "encodedinput.txt"


#hyperparameters
vocab_size = 1000 #amount of tokens in vocabulary

max_seq_size = 70 #sequence length
embedding_size = 512 #embedding (per vector size)
ffn_hidden = 512 #amount of hidden layers in feed-forward
num_heads = 8 #amount of attention heads
num_layers = 3 #amount of repetitions through heads
layer_size = 1
drop_prob = 0.25

batch_size = 16 #amount of samples (of tokens) per batch
learning_rate = 3e-5 #learning rate
epochs = 5 #epochs (repetitions through whole dataset)

#generation
amount_to_generate = 200
temperature = 1
top_k = 10