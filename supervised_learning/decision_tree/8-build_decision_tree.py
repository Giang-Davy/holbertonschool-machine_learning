#!/usr/bin/env python3
"""
Module implémentant les classes pour construire et manipuler un arbre de décision.
"""

import numpy as np
from sklearn import datasets

class Node:
    """
    Classe représentant un nœud dans un arbre de décision.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialise un nœud de l'arbre de décision.

        Args:
            feature (int, optional): L'indice de la caractéristique utilisée.
            threshold (float, optional): La valeur seuil pour la division.
            left_child (Node, optional): L'enfant gauche du nœud.
            right_child (Node, optional): L'enfant droit du nœud.
            is_root (bool, optional): Indique si le nœud est la racine.
            depth (int, optional): La profondeur du nœud dans l'arbre.
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
        self.indicator = None

    def max_depth_below(self):
        """
        Retourne la profondeur maximale de l'arbre sous ce nœud.
        """
        max_depth = self.depth
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Compte les nœuds dans le sous-arbre enraciné à ce nœud.
        Optionnellement, compte uniquement les feuilles.
        """
        if only_leaves:
            if self.is_leaf:
                return 1
            count = 0
        else:
            count = 1

        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne de caractères
        du nœud et de ses enfants.
        """
        node_type = "racine" if self.is_root else "nœud"
        details = (f"{node_type} [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            details += f"    +---> {left_str}"
        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            details += f"\n    +---> {right_str}"
        return details

    def get_leaves_below(self):
        """
        Retourne une liste de toutes les feuilles sous ce nœud.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Calcule récursivement, pour chaque nœud, deux dictionnaires stockés
        comme attributs Node.lower et Node.upper. Ces dictionnaires
        contiennent les limites pour chaque caractéristique.
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = max(
                self.threshold,
                self.left_child.lower.get(self.feature, -np.inf)
            )
            self.left_child.update_bounds_below()

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = min(
                self.threshold,
                self.right_child.upper.get(self.feature, np.inf)
            )
            self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Calcule la fonction indicatrice pour les bornes inférieures
        et supérieures pour chaque caractéristique.
        """
        def is_large_enough(x):
            return np.all(
                np.array([x[:, key] >= self.lower[key] for key in self.lower.keys()]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([x[:, key] <= self.upper[key] for key in self.upper.keys()]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )


class Leaf(Node):
    """
    Classe représentant une feuille dans un arbre de décision.
    """

    def __init__(self, value, depth=None):
        """
        Initialise une feuille avec une valeur et une profondeur.

        Args:
            value: La valeur de prédiction de la feuille.
            depth (int, optional): La profondeur de la feuille dans l'arbre.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille, car les feuilles
        sont les points finaux d'un arbre.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Retourne 1 car les feuilles comptent pour un nœud chacune.
        """
        return 1

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne de
        caractères de la feuille.
        """
        return f"-> feuille [value={self.value}] "

    def get_leaves_below(self):
        """
        Retourne une liste contenant uniquement cette feuille.
        """
        return [self]

    def update_bounds_below(self):
        """
        Les feuilles n'ont pas besoin de mettre à jour les limites car elles
        représentent les points finaux.
        """
        pass


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
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predictor = None

    def fit(self, explanatory, target, verbose=0):
        """
        Entraîne l'arbre de décision avec les données fournies.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(only_leaves=True)}
- Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")

    def depth(self):
        """
        Retourne la profondeur maximale de l'arbre.
        """
        return self._depth(self.root)

    def _depth(self, node):
        """
        Calcule récursivement la profondeur d'un nœud donné.
        """
        if isinstance(node, Leaf):
            return node.depth
        left_depth = self._depth(node.left_child) if node.left_child else 0
        right_depth = self._depth(node.right_child) if node.right_child else 0
        return max(left_depth, right_depth) + 1

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre de nœuds dans l'arbre, ou seulement les feuilles si only_leaves=True.
        """
        return self._count_nodes(self.root, only_leaves)

    def _count_nodes(self, node, only_leaves=False):
        """
        Calcule récursivement le nombre de nœuds (ou de feuilles) dans l'arbre.
        """
        if isinstance(node, Leaf):
            return 1 if only_leaves else 1
        left_count = self._count_nodes(node.left_child, only_leaves) if node.left_child else 0
        right_count = self._count_nodes(node.right_child, only_leaves) if node.right_child else 0
        return left_count + right_count + (0 if only_leaves else 1)

    def fit_node(self, node):
        """
        Entraîne un nœud de l'arbre de décision.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (self.explanatory[:, node.feature] <= node.threshold)
        right_population = node.sub_population & (self.explanatory[:, node.feature] > node.threshold)

        # Vérifie si le nœud gauche est une feuille
        is_left_leaf = self.is_leaf(left_population)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Vérifie si le nœud droit est une feuille
        is_right_leaf = self.is_leaf(right_population)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def is_leaf(self, sub_population):
        """
        Détermine si un nœud est une feuille en fonction des critères d'arrêt.
        """
        if np.sum(sub_population) < self.min_pop:
            return True
        if np.all(self.target[sub_population] == self.target[sub_population][0]):
            return True
        return False

    def random_split_criterion(self, node):
        """
        Divise le nœud de manière aléatoire en fonction d'une caractéristique.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def Gini_split_criterion(self, node):
        """
        Divise le nœud en fonction du critère de Gini.
        """
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(self.explanatory.shape[1]):
            feature_values = self.explanatory[:, feature][node.sub_population]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_population = node.sub_population & (feature_values <= threshold)
                right_population = node.sub_population & (feature_values > threshold)

                gini = self.gini_index(left_population) + self.gini_index(right_population)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def gini_index(self, population):
        """
        Calcule l'indice de Gini pour un ensemble de données donné.
        """
        if np.sum(population) == 0:
            return 0
        classes = np.unique(self.target[population])
        gini = 1
        for c in classes:
            p_c = np.sum(self.target[population] == c) / np.sum(population)
            gini -= p_c ** 2
        return gini

    def get_leaf_child(self, node, sub_population):
        """
        Crée un enfant feuille pour un nœud donné.
        """
        value = self.get_leaf_value(sub_population)
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Crée un enfant nœud pour un nœud donné.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def get_leaf_value(self, sub_population):
        """
        Retourne la valeur (classe) majoritaire pour un nœud feuille.
        """
        values, counts = np.unique(self.target[sub_population], return_counts=True)
        return values[np.argmax(counts)]

    def np_extrema(self, arr):
        """
        Retourne les bornes minimale et maximale d'un tableau numpy.
        """
        return np.min(arr), np.max(arr)

    def accuracy(self, test_explanatory, test_target):
        """
        Calcule l'exactitude du modèle.
        """
        return np.sum(np.equal(self.predict(test_explanatory), test_target)) / test_target.size

    def update_predict(self):
        """
        Met à jour la fonction de prédiction du modèle en fonction des données d'entraînement.
        """
        self.predictor = self.predict_node(self.root)

    def predict_node(self, node):
        """
        Prédit la classe pour un sous-ensemble de données donné.
        """
        if isinstance(node, Leaf):
            return node.value
        return lambda x: self.predict_node(node.left_child) if x[node.feature] <= node.threshold else self.predict_node(node.right_child)

    def predict(self, explanatory):
        """
        Prédit les classes pour un ensemble de données donné.
        """
        return np.array([self.predictor(x) for x in explanatory])
    
    def possible_thresholds(self, node, feature):
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        # Étape 1 : Calculer les seuils possibles
        thresholds = self.possible_thresholds(node, feature)

        # Étape 2 : Extraire les valeurs du feature pour la sous-population
        feature_values = self.explanatory[node.sub_population, feature]

        # Étape 3 : Initialiser une matrice booléenne indiquant l'appartenance aux classes
        classes = self.target[node.sub_population]
        unique_classes = np.unique(classes)
        class_matrix = np.array([classes == cls for cls in unique_classes]).T  # Shape (n, c)

        # Étape 4 : Calculer Left_F (shape (n, t, c))
        Left_F = feature_values[:, None] <= thresholds  # Shape (n, t)
        Left_F = Left_F[:, :, None] * class_matrix[:, None, :]  # Broadcast to (n, t, c)

        # Étape 5 : Calculer les impuretés Gini pour les enfants gauches
        left_counts = Left_F.sum(axis=0)  # Shape (t, c)
        left_totals = left_counts.sum(axis=1)  # Shape (t,)
        left_totals[left_totals == 0] = 1  # Éviter la division par zéro
        left_gini = 1 - (left_counts**2).sum(axis=1) / left_totals**2  # Shape (t,)

        # Étape 6 : Calculer les impuretés Gini pour les enfants droits
        total_class_counts = class_matrix.sum(axis=0)  # Shape (c,)
        right_counts = total_class_counts[None, :] - left_counts  # Shape (t, c)
        right_totals = right_counts.sum(axis=1)  # Shape (t,)
        right_totals[right_totals == 0] = 1  # Éviter la division par zéro
        right_gini = 1 - (right_counts**2).sum(axis=1) / right_totals**2  # Shape (t,)

        # Étape 7 : Calculer la Gini moyenne pour chaque seuil
        total_population = len(node.sub_population)
        gini_split = (
            (left_totals / total_population) * left_gini +
            (right_totals / total_population) * right_gini
        )  # Shape (t,)

        # Étape 8 : Trouver le seuil minimisant la Gini moyenne
        best_index = np.argmin(gini_split)
        best_threshold = thresholds[best_index]
        best_gini = gini_split[best_index]

        return best_threshold, best_gini
