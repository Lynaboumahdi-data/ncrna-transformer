import pandas as pd
import json
from collections import Counter
from tokenizer_kmers import kmer_tokenizer

k=3
counter = Counter()

df = pd.read_csv("dataset_final_ncRNA_clean.csv")

for seq in df["sequence"]:
    kmers = kmer_tokenizer(seq, k)
    counter.update(kmers)
vocab = {"<PAD>": 0 , "<UNK>":1}
for i, kmer in enumerate(counter.keys(), start=2):
    vocab[kmer] = i

print("vocab size : ",len(vocab))

with open("vocab_k3.json","w") as f:
    json.dump(vocab, f)