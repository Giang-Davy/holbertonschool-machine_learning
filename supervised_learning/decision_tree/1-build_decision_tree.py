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

    def count_nodes_below(self, only_leaves=False):
        """
        Compte le nombre de noeuds sous ce noeud, en excluant éventuellement
        les noeuds internes si `only_leaves` est True.

        Args:
            only_leaves (bool): Si True, compter uniquement les feuilles.

        Returns:
            int: Nombre de noeuds sous ce noeud.
        """
        if self.is_leaf:
            return 1 if only_leaves else 1

        node_count = 0
        if self.left_child:
            node_count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            node_count += self.right_child.count_nodes_below(only_leaves)

        if not only_leaves:
            node_count += 1

        return node_count

    def __str__(self):
        text = f"{'root' if self.is_root else 'node'} [feature={self.feature}, threshold={self.threshold}]"
        if self.left_child:
            text += "\n" + left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            text += "\n" + right_child_add_prefix(self.right_child.__str__())
        return text

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

    def count_nodes_below(self, only_leaves=False):
        """
        Returns: ff
        """
        return 1

    def __str__(self):
        return (f"-> leaf [value={self.value}]")

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

    def count_nodes(self, only_leaves=False):
        """
        Returns: ff
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return self.root.__str__()

def left_child_add_prefix(text):
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  " + x) + "\n"
    return new_text

def right_child_add_prefix(text):
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text
