dataset_name = "tinyshakespeare.txt"

vocab_size = 300 #amount of tokens in vocabulary

max_seq_size = 100 #sequence length
embedding_size = 512 #embedding (per vector size)
ffn_hidden = 2000 #amount of hidden layers in feed-forward
num_heads = 8 #amount of attention heads
num_layers = 3 #amount of repetitions through heads
layer_size = 1
drop_prob = 0.1

batch_size = 16 #amount of samples (of tokens) per batch
learning_rate = 3e-5 #learning rate
epochs = 10 #epochs (repetitions through whole dataset)