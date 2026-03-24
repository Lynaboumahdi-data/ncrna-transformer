# -*- coding: utf-8 -*-
"""
Fusion + échantillonnage des ncRNA
CSV léger + ID pour traçabilité
"""

import csv
import random

MAX_SEQ_PER_CLASS = 100_000
SAMPLING_RATE = 0.01  # ~1 %

files = {
    "../data/processed/lncRNA_clean.fa": "lncRNA",
    "../data/processed/RNAcentral_clean.fa": "RNAcentral",
    "../data/processed/MARS_clean.fa": "MARS"
}

output_file = "../data/processed/dataset_final_ncRNA_sampled_with_id.csv"

seq_id = 0

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "sequence", "label"])

    for fasta_file, label in files.items():
        print(f"Traitement {label}...")
        seq = ""
        count = 0

        with open(fasta_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith(">"):
                    if seq:
                        if random.random() < SAMPLING_RATE:
                            writer.writerow([f"seq_{seq_id}", seq, label])
                            seq_id += 1
                            count += 1
                            if count >= MAX_SEQ_PER_CLASS:
                                break
                        seq = ""
                else:
                    seq += line.strip().upper()

        print(f"{label} conservées : {count}")

print("✅ dataset_final_ncRNA_sampled_with_id.csv créé avec succès")
