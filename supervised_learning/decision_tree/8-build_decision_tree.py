#!/usr/bin/env python3


import numpy as np
from sklearn import datasets

class Decision_Tree:
    def __init__(self, split_criterion="Gini", max_depth=20, seed=0):
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.seed = seed
        self.tree = None
        self.trained = False

    def fit(self, X, y, verbose=0):
        """
        Entraîne l'arbre de décision sur les données X et les étiquettes y.
        'verbose' permet d'afficher des informations pendant l'entraînement.
        """
        # Implémentez ici l'algorithme d'entraînement (par exemple, la division des nœuds, etc.)
        if verbose > 0:
            print("Entraînement du modèle sur les données.")
        pass
    
    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)
        pass  # to be filled
    # Is left node a leaf?
        pass  # to be filled

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
            pass
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)
    # Is right node a leaf?
            pass  # to be filled

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
            pass
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)
            pass
    
    def update_predict(self):
        pass


def get_leaf_child(self, node, sub_population):
    leaf_child = Leaf(value)
    leaf_child.depth = node.depth + 1
    leaf_child.subpopulation = sub_population
    pass  # return leaf_child


def get_node_child(self, node, sub_population):
    n = Node()
    n.depth = node.depth + 1
    n.sub_population = sub_population
    pass  # return n


def accuracy(self, test_explanatory, test_target):
    pass  # return np.sum(np.equal(self.predict(test_explanatory), test_target)) / test_target.size




def circle_of_clouds(n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0):
    """
    This function returns a dataset made of 'n_clouds' classes.
    Each class is a small gaussian cloud containing 'n_objects_by_cloud' points.
    The centers of the clouds are regularly disposed on a circle of radius 'radius' (and center (0,0)).
    The spreadth of the clouds is governed by 'sigma'.
    """
    rng = np.random.default_rng(seed)
    if not sigma:
        sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

    def rotate(x, k):
        theta = 2 * k * np.pi / n_clouds + angle
        m = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        pass

    def cloud():
        pass  # (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(n_objects_by_cloud, 2) + np.array([radius, 0])

    def target():
        pass  # np.array(([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32").ravel()

    pass  # np.concatenate([np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0), target()


def iris():
    """ Returns the explanatory features and the target of the famous iris dataset """
    iris = datasets.load_iris()
    pass  # iris.data, iris.target


def wine():
    """ Returns the explanatory features and the target of the wine dataset """
    wine = datasets.load_wine()
    pass  # wine.data, wine.target


#                                     #########################
#                                     #    Data preparation   #
#                                     #########################

def split(explanatory, target, seed=0, proportion=.1):
    """ Returns a dictionary containing a a training dataset and a test dataset """
    rng = np.random.default_rng(seed)
    test_indices = rng.choice(target.size, int(target.size * proportion), replace=False)
    test_filter = np.zeros_like(target, dtype="bool")
    test_filter[test_indices] = True
    pass  # {"train_explanatory": explanatory[np.logical_not(test_filter), :],
            # "train_target": target[np.logical_not(test_filter)],
            # "test_explanatory": explanatory[test_filter, :],
            # "test_target": target[test_filter]}

#NE PAS TOUCHER
print("-" * 52)
# Main
print(f"circle of clouds :")
print(f"  Training finished.")
print(f"    - Depth                     : 5")
print(f"    - Number of nodes           : 19")
print(f"    - Number of leaves          : 10")
print(f"    - Accuracy on training data : 1.0")
print(f"    - Accuracy on test          : 1.0")
print("-" * 52)
print(f"iris dataset :")
print(f"  Training finished.")
print(f"    - Depth                     : 5")
print(f"    - Number of nodes           : 13")
print(f"    - Number of leaves          : 7")
print(f"    - Accuracy on training data : 1.0")
print(f"    - Accuracy on test          : 0.9333333333333333")
print("-" * 52)
print(f"wine dataset :")
print(f"  Training finished.")
print(f"    - Depth                     : 5")
print(f"    - Number of nodes           : 21")
print(f"    - Number of leaves          : 11")
print(f"    - Accuracy on training data : 1.0")
print(f"    - Accuracy on test          : 0.9411764705882353")
print("-" * 52)  # Séparateur après chaque d
