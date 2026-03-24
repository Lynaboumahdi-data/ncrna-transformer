def clean_sequence(seq):
    """
    Nettoyage des séquences ARN
    """
    seq = seq.upper().replace("T", "U")
    return "".join([c for c in seq if c in ["A", "U", "C", "G"]])


def split_kmers(seq, k=4):
    """
    Découpage en k-mers
    """
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]
