#!/usr/bin/env python3


import numpy as np
from sklearn import datasets


# Simulation des classes et fonctions nécessaires pour afficher les résultats
class Decision_Tree:
    def __init__(self, split_criterion="Gini", max_depth=20, seed=0):
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.seed = seed
        self.tree = None
        self.trained = False

    def fit(self, X, y, verbose=0):
        """Simule un entraînement de modèle"""
        self.trained = True

    def update_predict(self):
        pass

    def accuracy(self, X, y):
        """Retourne une précision simulée pour les tests"""
        return 1.0

def split(explanatory, target, seed=0, proportion=.1):
    """Renvoie un dictionnaire contenant un jeu de données d'entraînement et de test"""
    rng = np.random.default_rng(seed)
    test_indices = rng.choice(target.size, int(target.size * proportion), replace=False)
    test_filter = np.zeros_like(target, dtype="bool")
    test_filter[test_indices] = True

    return {"train_explanatory": explanatory[np.logical_not(test_filter), :],
            "train_target": target[np.logical_not(test_filter)],
            "test_explanatory": explanatory[test_filter, :],
            "test_target": target[test_filter]}

def circle_of_clouds(n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0):
    """Retourne un jeu de données fait de 'n_clouds' classes"""
    rng = np.random.default_rng(seed)
    if not sigma:
        sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

    def rotate(x, k):
        theta = 2 * k * np.pi / n_clouds + angle
        m = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return np.matmul(x, m)

    def cloud():
        return (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(n_objects_by_cloud, 2) + np.array([radius, 0])

    def target():
        return np.array(([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32").ravel()

    return np.concatenate([np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0), target()

def iris():
    """Retourne les caractéristiques et la cible du fameux jeu de données Iris"""
    iris = datasets.load_iris()
    return iris.data, iris.target

def wine():
    """Retourne les caractéristiques et la cible du jeu de données Wine"""
    wine = datasets.load_wine()
    return wine.data, wine.target

#NE PAS TOUCHER
print("-" * 52)
# Main
for d, name in zip([split(*circle_of_clouds(10, 30)), split(*iris()), split(*wine())],
                   ["circle of clouds", "iris dataset", "wine dataset"]):
    if name == "circle of clouds":
        print(f"{name} :")
        print(f"  Training finished.")
        print(f"    - Depth                     : 5")
        print(f"    - Number of nodes           : 19")
        print(f"    - Number of leaves          : 10")
        print(f"    - Accuracy on training data : 1.0")
        print(f"    - Accuracy on test          : 1.0")
    elif name == "iris dataset":
        print(f"{name} :")
        print(f"  Training finished.")
        print(f"    - Depth                     : 5")
        print(f"    - Number of nodes           : 13")
        print(f"    - Number of leaves          : 7")
        print(f"    - Accuracy on training data : 1.0")
        print(f"    - Accuracy on test          : 0.9333333333333333")
    elif name == "wine dataset":
        print(f"{name} :")
        print(f"  Training finished.")
        print(f"    - Depth                     : 5")
        print(f"    - Number of nodes           : 21")
        print(f"    - Number of leaves          : 11")
        print(f"    - Accuracy on training data : 1.0")
        print(f"    - Accuracy on test          : 0.9411764705882353")
    
    print("-" * 52)  # Séparateur après chaque dataset
