def load_fasta(file_path):
    """
    Lecture d'un fichier FASTA
    """
    sequences = []
    seq = ""

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.strip()

        if seq:
            sequences.append(seq)

    return sequences
