from gensim.models import Word2Vec
from tqdm import tqdm

def get_kmers(sequence, k=4):
    """Découpe une séquence en k-mers"""
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

def compute_embeddings(sequences, k=4, vector_size=100, window=5, min_count=1, workers=4):
    """
    Génère des embeddings Word2Vec pour une liste de séquences.
    """
    if not sequences:
        raise ValueError("Aucune séquence valide pour générer des embeddings.")

    print(f"Découpage des séquences en k-mers (k={k})...")
    kmer_sequences = [get_kmers(seq, k) for seq in tqdm(sequences, desc="Découpage")]

    print(" Entraînement du modèle Word2Vec...")
    model = Word2Vec(
        sentences=kmer_sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1  
    )

    print("Calcul des embeddings moyens par séquence...")
    embeddings = []
    for kmers in tqdm(kmer_sequences, desc="Embeddings"):
        vecs = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
        if vecs:
            mean_vec = sum(vecs) / len(vecs)
        else:
            mean_vec = [0.0]*vector_size
        embeddings.append(mean_vec)

    import numpy as np
    return np.array(embeddings)
