#!/usr/bin/env python3
""" Tâche 10"""


import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """
    La classe Isolation_Random_Tree implémente un
    arbre d'isolation pour détecter les valeurs aberrantes.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
        args: ff
        returns:
            None
        """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        args:
            None
        returns:
            str: La représentation en chaîne de l'arbre de décision.
        """
        return self.root.__str__() + "\n"

    def depth(self):
        """
        args:
            None
        returns:
            int: La profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        args:
            only_leaves (bool): Si True, compte uniquement les feuilles.
        returns:
            int: Le nombre de noeuds dans l'arbre.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        args:
            None
        returns:
            None
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        args:
            None
        returns:
            list: La liste de toutes les feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        args:
            None
        returns:
            None
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            """
            args:
                A (np.ndarray): Un tableau NumPy 2D de forme
                                 (n_individuals, n_features), où chaque ligne
                                 représente un individu avec ses caractéristiques.
            returns:
                np.ndarray: Un tableau NumPy 1D de forme (n_individuals, ),
                            où chaque élément est la classe prédite
                            pour l'individu correspondant dans A.
            """
            predictions = np.zeros(A.shape[0], dtype=int)
            for i, x in enumerate(A):
                for leaf in leaves:
                    if leaf.indicator(np.array([x])):
                        predictions[i] = leaf.value
                        break
            return predictions
        self.predict = predict

    def np_extrema(self, arr):
        """
        args:
            arr (similaire à un tableau): Tableau dont on veut trouver les extrema.
        returns:
            tuple: Valeurs minimale et maximale du tableau.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        args:
            node (Node): Le noeud à traiter pour la séparation.
        returns:
            tuple: L'indice de la caractéristique choisie et la valeur seuil.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                    self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        args:
            node (Node): Le noeud parent.
            sub_population (similaire à un tableau): La sous-population associée.
        returns:
            Leaf: La feuille enfant.
        """
        value = node.depth + 1
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        args:
            node (Node): Le noeud parent.
            sub_population (similaire à un tableau): La sous-population associée.
        returns:
            Node: Le noeud enfant non-feuille créé.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        args:
            node (Node): Le noeud à ajuster.
        returns:
            None
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & \
                (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population

        is_left_leaf = (node.depth == self.max_depth - 1 or
                                            np.sum(left_population) <= self.min_pop)
        is_right_leaf = (node.depth == self.max_depth - 1 or
                                                 np.sum(right_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.depth = node.depth + 1
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.depth = node.depth + 1
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        args: ff
        returns:
            None
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Entraînement terminé.
    - Profondeur                : {self.depth()}
    - Nombre de noeuds          : {self.count_nodes()}
    - Nombre de feuilles        : {self.count_nodes(only_leaves=True)}""")
