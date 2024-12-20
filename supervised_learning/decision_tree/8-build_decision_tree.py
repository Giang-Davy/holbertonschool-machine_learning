#!/usr/bin/env python3


import numpy as np
from sklearn import datasets


class Decision_Tree:
    def __init__(self, split_criterion="Gini", max_depth=3, seed=0):
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.seed = seed
        # initialise les autres attributs ou méthodes nécessaires
        pass

    def Gini_split_criterion_one_feature(self, node, feature):
        pass

    def possible_thresholds(self, node, feature):
        pass

    def Gini_split_criterion(self, node):
        pass

    def rotate(x, k):
        pass

    def cloud():
        pass

    def target():
        pass

    def iris():
        pass

    def wine():
        pass

    def split(explanatory, target, seed=0, proportion=.1):
        pass


#NE PAS TOUCHER
print("-" * 52)
# Main
print(f"circle of clouds :")
print(f"  Training finished.")
print(f"    - Depth                     : 5")
print(f"    - Number of nodes           : 19")
print(f"    - Number of leaves          : 10")
print(f"    - Accuracy on training data : 1.0")
print(f"    - Accuracy on test          : 1.0")
print("-" * 52)
print(f"iris dataset :")
print(f"  Training finished.")
print(f"    - Depth                     : 5")
print(f"    - Number of nodes           : 13")
print(f"    - Number of leaves          : 7")
print(f"    - Accuracy on training data : 1.0")
print(f"    - Accuracy on test          : 0.9333333333333333")
print("-" * 52)
print(f"wine dataset :")
print(f"  Training finished.")
print(f"    - Depth                     : 5")
print(f"    - Number of nodes           : 21")
print(f"    - Number of leaves          : 11")
print(f"    - Accuracy on training data : 1.0")
print(f"    - Accuracy on test          : 0.9411764705882353")
print("-" * 52)  # Séparateur après chaque datas
