def kmer_tokenizer(sequence,k=3):
    sequence = sequence.upper().replace("T","U")
    return [sequence[i:i+k] for i in range(len(sequence) - k +1)]