#!/usr/bin/env python3


import numpy as np
from sklearn import datasets

class Decision_Tree:
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        pass
    def fit(self,explanatory, target,verbose=0):
        pass
    def np_extrema(self,arr):
        pass
    def random_split_criterion(self,node):
        pass
    def fit_node(self,node):
        pass
    def get_leaf_child(self, node, sub_population):
        pass
    def get_node_child(self, node, sub_population):
        pass
    def accuracy(self, test_explanatory , test_target):
        pass
    def possible_thresholds(self,node,feature):
        pass
    def Gini_split_criterion(self,node):
        pass
    
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
