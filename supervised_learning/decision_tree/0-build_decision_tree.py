#!/usr/bin/env python3
"""fonction"""


import numpy as np


class Node:
    """
    Noeud interne de l'arbre de décision.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialise un noeud interne.

        Args:
            feature (int, optional): Caractéristique pour la séparation.
            threshold (float, optional): Seuil de séparation.
            left_child (Node or Leaf, optional): Sous-arbre gauche.
            right_child (Node or Leaf, optional): Sous-arbre droit.
            is_root (bool, optional): Noeud racine (par défaut False).
            depth (int, optional): Profondeur du noeud (par défaut 0).
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calcule la profondeur maximale sous ce noeud.

        Returns:
            int: Profondeur maximale sous ce noeud.
        """
        if self.is_leaf:
            return self.depth
        left_depth = (self.left_child.max_depth_below()
                      if self.left_child else self.depth)
        right_depth = (self.right_child.max_depth_below()
                       if self.right_child else self.depth)
        return max(left_depth, right_depth)


class Leaf(Node):
    """
    Représente une feuille dans un arbre de décision.
    """

    def __init__(self, value, depth=None):
        """
        Initialise une feuille.

        Args:
            value (any): Valeur associée à cette feuille.
            depth (int, optional): Profondeur de la feuille.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille.

        Returns:
            int: La profondeur de la feuille.
        """
        return self.depth


class Decision_Tree:
    """
    Représente un arbre de décision.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise un arbre de décision.

        Args:
            max_depth (int, optional): Profondeur maximale de l'arbre.
            min_pop (int, optional): Taille minimale des sous-populations.
            seed (int, optional): Graine pour le générateur.
            split_criterion (str, optional): Critère de séparation.
            root (Node, optional): Racine de l'arbre.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Calcule la profondeur maximale de l'arbre.

        Returns:
            int: Profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()
