# =====================================
# Task 2 – Déséquilibrage des classes
# =====================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

# =====================================
# Chemins
# =====================================
CSV_PATH = r"..\data\processed\dataset_final_ncRNA_clean.csv"
OUTPUT_DIR = r"..\data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# Colonnes de csv
# =====================================
ID_COL = "id_seq"
SEQ_COL = "sequence"
LABEL_COL = "label"

# =====================================
# 1. Chargement des données
# =====================================
data = pd.read_csv(
    CSV_PATH,
    usecols=[ID_COL, SEQ_COL, LABEL_COL]
)

print("Dataset chargé :", data.shape)
print("Colonnes :", data.columns.tolist())

# =====================================
# 2. Analyse du déséquilibre des classes
# =====================================
label_dist = data[LABEL_COL].value_counts()
print("\nDistribution des classes :")
print(label_dist)

# Sauvegarde distribution
label_dist.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"))

# Graphique
plt.figure(figsize=(7, 4))
label_dist.plot(kind="bar")
plt.title("Répartition des classes ncRNA")
plt.xlabel("Classe")
plt.ylabel("Nombre de séquences")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution.png"),
    dpi=300
)
plt.close()

# =====================================
# 3. Encodage k-mers (TF-IDF)
# =====================================
def seq_to_kmers(seq, k=6):
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))

print("\nEncodage k-mers...")
data["kmers"] = data[SEQ_COL].apply(seq_to_kmers)

vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=10,
    sublinear_tf=True
)

X = vectorizer.fit_transform(data["kmers"])

# =====================================
# 4. Encodage labels + split (traçable)
# =====================================
encoder = LabelEncoder()
y = encoder.fit_transform(data[LABEL_COL])

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    data.index,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================
# 5. Baseline 
# =====================================
clf_base = LogisticRegression(max_iter=1000)
clf_base.fit(X_train, y_train)

pred_base = clf_base.predict(X_test)
acc_base = accuracy_score(y_test, pred_base)
f1_base = f1_score(y_test, pred_base, average="macro")

print("\nBaseline")
print("Accuracy :", acc_base)
print("F1-macro :", f1_base)

# =====================================
# 6. Pondération des classes
# =====================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

weights_dict = dict(enumerate(class_weights))

clf_weighted = LogisticRegression(
    max_iter=1000,
    class_weight=weights_dict
)

clf_weighted.fit(X_train, y_train)
pred_weighted = clf_weighted.predict(X_test)

acc_weighted = accuracy_score(y_test, pred_weighted)
f1_weighted = f1_score(y_test, pred_weighted, average="macro")

print("\nClass weights")
print("Accuracy :", acc_weighted)
print("F1-macro :", f1_weighted)

# =====================================
# 7. Sauvegarde des prédictions AVEC id_seq
# =====================================
predictions_df = pd.DataFrame({
    "id_seq": data.loc[idx_test, ID_COL],
    "true_label": encoder.inverse_transform(y_test),
    "predicted_label": encoder.inverse_transform(pred_weighted),
    "method": "class_weights"
})
predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions_with_ids.csv"),
    index=False
)

# =====================================
# 8. Synthèse finale
# =====================================
summary = pd.DataFrame({
    "Méthode": ["Baseline", "Class weights"],
    "Accuracy": [acc_base, acc_weighted],
    "F1-macro": [f1_base, f1_weighted]
})

summary.to_csv(
    os.path.join(OUTPUT_DIR, "imbalance_metrics.csv"),
    index=False
)

print("\nComparaison finale :")
print(summary)

print("\n✔ Task 2 terminée – pipeline conforme et traçable")