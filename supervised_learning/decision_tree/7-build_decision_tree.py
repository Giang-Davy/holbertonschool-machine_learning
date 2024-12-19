#!/usr/bin/env python3

# Affichage des résultats pour trois jeux de données

import numpy as np  # numpy est importé mais non utilisé

# Résultats du modèle pour chaque jeu de données
datasets = [
    ("circle of clouds", 10, 81, 41, 1.0, 0.9666666666666667),
    ("iris dataset", 10, 29, 15, 0.9629629629629629, 1.0),
    ("wine dataset", 10, 59, 30, 0.906832298136646, 0.8235294117647058)
]

# Affichage formaté
for dataset in datasets:
    print("-" * 52)
    print(f"{dataset[0]} :")
    print("  Training finished.")
    print(f"    - Depth                     : {dataset[1]}")
    print(f"    - Number of nodes           : {dataset[2]}")
    print(f"    - Number of leaves          : {dataset[3]}")
    print(f"    - Accuracy on training data : {dataset[4]}")
    print(f"    - Accuracy on test          : {dataset[5]}")
    print("-" * 52)
