#!/usr/bin/env python3
""" Task 11: 11. IRF 2 : isolation random forests """

import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree

class Isolation_Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
            Random seed for reproducibility (default is 0).
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts outlier scores for the given explanatory
            The average outlier score across all trees for each data point.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Trains the Isolation Random Forest on the given
        explanatory variables.
        None
        """
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
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory
        that have the smallest mean depth.

        Parameters:
        explanatory : numpy.ndarray
            The dataset of explanatory variables.
        Returns:
        numpy.ndarray
            corresponding to the n_suspects
            with the smallest mean depth.
        """
        depths = self.predict(explanatory)
        suspect_indices = np.argsort(depths)[:n_suspects]
        
        # Affichage formaté pour les suspects et leurs profondeurs
        print(f"suspects : {explanatory[suspect_indices]}")
        print(f"depths of suspects : {depths[suspect_indices]}")
        
        return explanatory[suspect_indices], depths[suspect_indices]
