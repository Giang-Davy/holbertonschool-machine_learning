#!/usr/bin/env python3
"""fonction"""


import numpy as np


class Node:
    """
    Nœud dans un arbre de décision.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialise un nœud.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = {}
        self.upper = {}
        if feature is not None:
            self.lower[feature] = -100
            self.upper[feature] = 100

    def __str__(self):
        """
        Représentation en chaîne du nœud.
        """
        node_type = "root" if self.is_root else "node"
        details = f"{node_type} [feature={self.feature},"
        details += f"threshold={self.threshold}]\n"
        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            details += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            details += f"\n    +---> {right_str}"

        return details.rstrip()


class Leaf(Node):
    """
    Feuille dans un arbre de décision.
    """

    def __init__(self, value, depth=None):
        """
        Initialise une feuille.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Représentation en chaîne de la feuille.
        """
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """
    Arbre de décision.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """
        Initialise l'arbre de décision.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

    def __str__(self):
        """
        Représentation en chaîne de l'arbre de décision.
        """
        return self.root.__str__()

    def update_predict(self):
        """
        Méthode de mise à jour des prédictions (non implémentée).
        """
        pass

    def update_bounds(self):
        """
        Met à jour les bornes des nœuds.
        """
        def recursive_update(node):
            if node.is_leaf:
                return
            if node.feature not in node.lower:
                node.lower[node.feature] = -100
            if node.feature not in node.upper:
                node.upper[node.feature] = 100
            node.lower[node.feature] = min(
                    node.lower[node.feature], node.threshold)
            node.upper[node.feature] = max(
                    node.upper[node.feature], node.threshold)
            recursive_update(node.left_child)
            recursive_update(node.right_child)

        recursive_update(self.root)

    def pred(self, sample):
        """
        Prédit la classe d'un échantillon.
        """
        return self._predict_sample(sample, self.root)

    def _predict_sample(self, sample, node):
        """
        Prédit la classe pour un échantillon donné à partir d'un nœud.
        """
        if node.is_leaf:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left_child)
        else:
            return self._predict_sample(sample, node.right_child)

    def predict(self, X):
        """
        Prédit les classes pour un ensemble d'échantillons.
        """
        predictions = [
            self._predict_sample(sample, self.root) for sample in X
        ]
        return np.array(predictions)

    def get_leaves(self):
        """
        Récupère les feuilles de l'arbre.
        """
        leaves = []

        def recursive_collect_leaves(node):
            if node.is_leaf:
                leaves.append(node)
                return
            recursive_collect_leaves(node.left_child)
            recursive_collect_leaves(node.right_child)

        recursive_collect_leaves(self.root)
        return leaves
