#!/usr/bin/env python3
""" Task 11: 11. IRF 2 : isolation random forests """

import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree

class Isolation_Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    """
    Initialise les paramètres de l'instance
    """

    def predict(self, explanatory):
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    """
    Prédit les scores d'outliers moyens pour les données
    """

    def fit(self, explanatory, n_trees=100, verbose=0):
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth, seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        
        self.target = self.predict(explanatory)
        
        if verbose == 1:
            print(f"----------------------------------------------------")
            print(f"circle of clouds :")
            print(f"  Training finished.")
            print(f"    - Mean depth                     : {np.array(depths).mean():.2f}")
            print(f"    - Mean number of nodes           : {np.array(nodes).mean():.2f}")
            print(f"    - Mean number of leaves          : {np.array(leaves).mean():.2f}")

    """
    Entraîne le modèle sur les données d'explanatoires
    """

    def suspects(self, explanatory, n_suspects):
        depths = self.predict(explanatory)
        suspect_indices = np.argsort(depths)[:n_suspects]
        suspects_data = explanatory[suspect_indices]
        suspect_depths = depths[suspect_indices]
        
        print(f"suspects : {suspects_data}")
        print(f"depths of suspects : {suspect_depths}")
        
        return suspects_data, suspect_depths

    """
    Retourne les n_suspects avec les plus petites profondeurs moyennes
    """

