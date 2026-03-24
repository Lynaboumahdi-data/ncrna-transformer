# preprocess_tokens.py
import pandas as pd
import numpy as np
import ast
import tensorflow as tf

# Load tokenized data
DATA_PATH = r"C:\Users\mathc\Desktop\aprentissage\New folder\Projet_rnaa (4)\Projet_rnaa (4)\Projet_rnaa\step3\dataset_ncRNA_encoded_k3.csv"
df = pd.read_csv(DATA_PATH)

# Convert string "[2, 3, 4, ...]" to actual Python list
input_ids = df['input_ids'].apply(ast.literal_eval)

# Pad to uniform length (512 tokens)
MAX_LEN = 512
X = tf.keras.preprocessing.sequence.pad_sequences(
    input_ids, maxlen=MAX_LEN, padding='post', truncating='post'
)

# Extract labels
y = df['label'].values

# Save processed data
np.save('X_token_matrix.npy', X)  # Shape: (n_samples, 512)
np.save('y_labels.npy', y)        # Shape: (n_samples,)

print(f"✅ Processed X shape: {X.shape}")
print(f"✅ Labels: {np.unique(y, return_counts=True)}")