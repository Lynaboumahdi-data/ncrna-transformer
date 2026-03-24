
import torch
import torch.nn as nn 
import json

with open("vocab_k3.json", "r") as f:
    vocab = json.load(f)


VOCAB_SIZE = len(vocab)
EMBED_DIM = 256
embedding = nn.Embedding(
   num_embeddings = VOCAB_SIZE,
   embedding_dim = EMBED_DIM,
   padding_idx = 0
   )
sample_input = torch.tensor([12,45,33,0,0])
output = embedding(sample_input)
print(output)