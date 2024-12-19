#!/usr/bin/env python3
"""fonction"""

import numpy as np
from sklearn import datasets

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
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.depth = 0
        self.num_nodes = 0
        self.num_leaves = 0
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

    def grow_tree(self, X, y, depth):
        """
        Méthode pour construire l'arbre de décision.
        """
        if depth >= self.max_depth or len(y) <= 1:
            return Leaf(value=np.bincount(y).argmax(), depth=depth)
        
        feature = self.rng.choice(X.shape[1])
        threshold = np.median(X[:, feature])
        
        X_left, y_left, X_right, y_right = self.split_data(X, y, feature, threshold)
        
        if len(y_left) == 0 or len(y_right) == 0:
            return Leaf(value=np.bincount(y).argmax(), depth=depth)
        
        left_child = self.grow_tree(X_left, y_left, depth + 1)
        right_child = self.grow_tree(X_right, y_right, depth + 1)
        
        return Node(feature=feature, threshold=threshold, left_child=left_child, right_child=right_child, depth=depth)

    def split_data(self, X, y, feature, threshold):
        """
        Divise les données en fonction de la caractéristique et du seuil.
        """
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    def fit(self, explanatory, target, X_test, y_test, verbose=0):
        """
        Entraîne l'arbre de décision et affiche l'accuracy sur les données de test.
        """
        self.root = self.grow_tree(explanatory, target, depth=0)

        if verbose > 0:
            print("Training finished.")
            print(f"    - Depth                     : {self.max_depth}")
            print(f"    - Number of nodes           : {self.count_nodes(self.root)}")
            print(f"    - Number of leaves          : {self.count_leaves(self.root)}")
            print(f"    - Accuracy on training data : {self.accuracy(explanatory, target)}")
        
        print(f"    - Accuracy on test          : {self.accuracy(X_test, y_test)}")

    def accuracy(self, X, y):
        """
        Calcule l'accuracy du modèle sur les données X et leurs cibles y.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée X.
        """
        def traverse(node, x):
            if isinstance(node, Leaf):
                return node.value
            if x[node.feature] <= node.threshold:
                return traverse(node.left_child, x)
            else:
                return traverse(node.right_child, x)

        return np.array([traverse(self.root, x) for x in X])

    def count_nodes(self, node):
        """
        Compte le nombre de nœuds dans l'arbre.
        """
        if isinstance(node, Leaf):
            return 1
        return 1 + self.count_nodes(node.left_child) + self.count_nodes(node.right_child)

    def count_leaves(self, node):
        """
        Compte le nombre de feuilles dans l'arbre.
        """
        if isinstance(node, Leaf):
            return 1
        return self.count_leaves(node.left_child) + self.count_leaves(node.right_child)
