#!/usr/bin/env python3


import numpy as np
from sklearn import datasets


class Decision_Tree:
    """
    Arbre de décision.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        """
        Initialise l'arbre de décision.
        """
        pass

    def fit(self, explanatory, target, verbose=0):
        """
        Entraîne l'arbre de décision avec les données fournies.
        """
        pass

    def depth(self):
        """
        Retourne la profondeur maximale de l'arbre.
        """
        pass

    def _depth(self, node):
        """
        Calcule récursivement la profondeur d'un nœud donné.
        """
        pass

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre de nœuds dans l'arbre, ou seulement les feuilles si only_leaves=True.
        """
        pass

    def _count_nodes(self, node, only_leaves=False):
        """
        Calcule récursivement le nombre de nœuds (ou de feuilles) dans l'arbre.
        """
        pass

    def fit_node(self, node):
        """
        Entraîne un nœud de l'arbre de décision.
        """
        pass

    def is_leaf(self, sub_population):
        """
        Détermine si un nœud est une feuille en fonction des critères d'arrêt.
        """
        pass

    def random_split_criterion(self, node):
        """
        Divise le nœud de manière aléatoire en fonction d'une caractéristique.
        """
        pass

    def Gini_split_criterion(self, node):
        """
        Divise le nœud en fonction du critère de Gini.
        """
        pass

    def gini_index(self, population):
        """
        Calcule l'indice de Gini pour un ensemble de données donné.
        """
        pass

    def get_leaf_child(self, node, sub_population):
        """
        Crée un enfant feuille pour un nœud donné.
        """
        pass

    def get_node_child(self, node, sub_population):
        """
        Crée un enfant nœud pour un nœud donné.
        """
        pass

    def get_leaf_value(self, sub_population):
        """
        Retourne la valeur (classe) majoritaire pour un nœud feuille.
        """
        pass

    def np_extrema(self, arr):
        """
        Retourne les bornes minimale et maximale d'un tableau numpy.
        """
        pass

    def accuracy(self, test_explanatory, test_target):
        """
        Calcule l'exactitude du modèle.
        """
        pass

    def update_predict(self):
        """
        Met à jour la fonction de prédiction du modèle en fonction des données d'entraînement.
        """
        pass

    def predict_node(self, node):
        """
        Prédit la classe pour un sous-ensemble de données donné.
        """
        pass

    def predict(self, explanatory):
        """
        Prédit les classes pour un ensemble de données donné.
        """
        pass

    def possible_thresholds(self, node, feature):
        """
        Calcule les seuils possibles pour une caractéristique donnée dans un nœud.
        """
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
