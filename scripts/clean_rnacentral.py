# -*- coding: utf-8 -*-
"""
Nettoyage RNAcentral
"""

from Bio import SeqIO
import os

MIN_LEN = 50
MAX_LEN = 10000

input_fasta = r"C:\Users\LENOVO\Desktop\projet_rna\data\raw\rnacentral_active.fasta"
output_fasta = r"C:\Users\LENOVO\Desktop\projet_rna\data\processed\RNAcentral_clean.fa"

os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

total = 0
kept = 0

with open(output_fasta, "w", encoding="utf-8") as out_handle:
    for record in SeqIO.parse(input_fasta, "fasta"):
        total += 1

        seq = str(record.seq).upper()
        seq = seq.replace("T", "U")
        seq = "".join(c for c in seq if c in "AUGC")

        if MIN_LEN <= len(seq) <= MAX_LEN:
            out_handle.write(f">{record.id}\n")
            out_handle.write(seq + "\n")
            kept += 1

print("RNAcentral total :", total)
print("RNAcentral gardées :", kept)
print("RNAcentral supprimées :", total - kept)
print("✔ RNAcentral_clean.fa créé")
