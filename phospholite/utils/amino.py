from typing import Dict


standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY" 

amino_acid_names = {
    "A": "Alanine",
    "C": "Cysteine",
    "D": "Aspartic acid",
    "E": "Glutamic acid",
    "F": "Phenylalanine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "K": "Lysine",
    "L": "Leucine",
    "M": "Methionine",
    "N": "Asparagine",
    "P": "Proline",
    "Q": "Glutamine",
    "R": "Arginine",
    "S": "Serine",
    "T": "Threonine",
    "V": "Valine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
}

STANDARD_AMINO_ACID_MAPPING_3_TO_1: Dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PYL": "O",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "SEC": "U",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}