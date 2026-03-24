import pandas as pd
import numpy as np
from embedding import compute_embeddings
DATA_PATH = "C:\\Users\\High Tech\\Downloads\\Telegram Desktop\\Projet_rnaa\\Projet_rnaa\\data\\processed\\dataset_final_ncRNA_clean.csv"

print("Chargement des séquences depuis le CSV...")
df = pd.read_csv(DATA_PATH)
if 'sequence' not in df.columns or 'label' not in df.columns:
    raise ValueError("Le CSV doit contenir les colonnes 'sequence' et 'label'.")

sequences = df['sequence'].tolist()
labels = df['label'].tolist()

print("Nettoyage des séquences...")
def clean_sequence(seq):
    return ''.join([c for c in seq.upper() if c in 'ACGU'])

sequences = [clean_sequence(seq) for seq in sequences]
sequences = [seq for seq in sequences if len(seq) >= 4]

print(f" Génération des embeddings Word2Vec pour {len(sequences)} séquences...")
X = compute_embeddings(sequences, k=4, vector_size=100, workers=4)
np.save("embeddings_task4.npy", X)
np.save("labels_task4.npy", np.array(labels[:len(sequences)]))  # attention à garder la même taille
print("Nombre de séquences :", X.shape[0])
print("Dimension des embeddings :", X.shape[1])
