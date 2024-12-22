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
        max_depth : int, optionnel
                Profondeur maximale de l'arbre (par défaut 10).
        seed : int, optionnel
                Graine pour la génération de nombres aléatoires (par défaut 0).
        root : Node ou Leaf, optionnel
        """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Retourne une représentation en chaîne de l'arbre de décision.

        Retourne :
        str
                La représentation en chaîne de l'arbre de décision.
        """
        return self.root.__str__() + "\n"

    def depth(self):
        """
        Retourne la profondeur maximale de l'arbre.

        Retourne :
        int
                La profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre de noeuds dans l'arbre de décision.

        Paramètres :
        only_leaves : bool, optionnel

        Retourne :
        int
                Le nombre de noeuds dans l'arbre.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Met à jour les limites pour l'ensemble
        de l'arbre en partant du noeud racine.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Retourne une liste de toutes les feuilles de l'arbre.

        Retourne :
        list
                La liste de toutes les feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Met à jour la fonction de prédiction pour l'arbre de décision.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            """
            Prédit la classe pour chaque individu dans le
            tableau d'entrée A en utilisant l'arbre de décision.

            Paramètres :
            A : np.ndarray
                    Un tableau NumPy 2D de forme (n_individuals,
                    n_features), où chaque ligne
                    représente un individu avec ses caractéristiques.

            Retourne :
            np.ndarray
                    Un tableau NumPy 1D de forme (n_individuals,),
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
        Retourne les valeurs minimale et maximale d'un tableau.

        Paramètres :
        arr : similaire à un tableau
                Tableau dont on veut trouver les extrema.

        Retourne :
        tuple
                Valeurs minimale et maximale du tableau.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """

        Paramètres
        node : Node

        Retourne
        tuple
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

        Paramètres :
        node : Node
                Le noeud parent.
        sub_population : similaire à un tableau

        Retourne :
        Leaf
        """
        value = node.depth + 1
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Crée un noeud enfant non-feuille.

        Paramètres
        node : Node
                Le noeud parent.
        sub_population : similaire à un tableau
                La sous-population pour le noeud enfant.

        Retourne :
        Node
                Le noeud enfant non-feuille créé.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """

        Paramètres :
        node : Node
                Le noeud à ajuster.
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

        Paramètres :
        explanatory : similaire à un tableau
                Variables explicatives utilisées pour l'entraînement.
        verbose : int, optionnel
                Si défini à 1, imprime les statistiques de l'entraînement (par défaut 0).
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
