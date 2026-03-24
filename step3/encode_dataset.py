import pandas as pd
import json
from tokenizer_kmers import kmer_tokenizer

MAX_LEN = 500 
PAD_ID,UNK_ID = 0,1
def encode_sequence(seq, vocab, k=3):
    kmers= kmer_tokenizer(seq, k)
    encoded = [vocab.get(km, UNK_ID) for km in kmers]
    if len(encoded) < MAX_LEN:
        encoded += [PAD_ID] * (MAX_LEN - len(encoded))
    else : 
        encoded = encoded[:MAX_LEN]
    return encoded

if __name__ == "__main__" : 
    df = pd.read_csv("dataset_final_ncRNA_clean.csv")
    label_map = {
       "lncRNA":0,
       "RNAcentral":1,
       "MARS":2
       }
    df["label"] = df["label"].map(label_map)
    with open("vocab_k3.json","r") as f:
        vocab = json.load(f)
    df["input_ids"] = df["sequence"].apply(
        lambda x: encode_sequence(x, vocab))
    df_encoded = df[["id_seq","input_ids","label"]]
    df_encoded.to_csv("dataset_ncRNA_encoded_k3.csv",index=False)
    
