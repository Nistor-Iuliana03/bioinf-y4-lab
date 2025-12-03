"""
Exercitiul 5 — Construirea unui arbore Neighbor-Joining (completat)

Pasii urmati:
1. Incarcam multi-FASTA-ul din data/work/<handle>/lab04/
2. Calculam matricea de distante (identity)
3. Construim arborele NJ
4. Salvam arborele in format Newick (.nwk)
"""

from pathlib import Path
from Bio import AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

HANDLE = "Nistor-Iuliana03"

FASTA = Path(f"../../../../data/work/{HANDLE}/lab04/your_sequences.fasta")
OUT_TREE = Path(f"tree_{HANDLE}.nwk")


if __name__ == "__main__":

    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA not found at {FASTA}")

    alignment = AlignIO.read(FASTA, "fasta")
    print(f"Loaded {len(alignment)} sequences.")

    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(alignment)
    print("\nDistance matrix:")
    print(dm)

    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    tree.root_at_midpoint()

    Phylo.write(tree, OUT_TREE, "newick")
    print(f"\nSaved NJ tree to: {OUT_TREE}\n")

    print("ASCII tree:")
    Phylo.draw_ascii(tree)
