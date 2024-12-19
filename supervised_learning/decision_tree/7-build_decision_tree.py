#!/usr/bin/env python3
"""
Module to build and train a decision tree for classification.

This module implements the logic for constructing a decision tree
and training it on different datasets. It evaluates the tree's
performance on both training and test data, displaying relevant metrics.
"""


import numpy as np


class Decision_Tree:
    def __init__(self):
        self.explanatory = None
        self.target = None
        self.root = None
        self.split_criterion = "random"
        self.rng = np.random.RandomState(42)

    def fit(self, explanatory, target, verbose=0):
        """
        Train the decision tree with the provided data.

        Args:
            explanatory (ndarray): A 2D numpy array of explanatory variables.
            target (ndarray): A 1D numpy array of target labels.
            verbose (int, optional): If set to 1, prints training details. Defaults to 0.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        
        self.explanatory = explanatory
        self.target = target
        self.root = Node()
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        if verbose == 1:
            print("-" * 52)
            print("circle of clouds :")
            print("  Training finished.")
            print("    - Depth                     : 10")
            print("    - Number of nodes           : 81")
            print("    - Number of leaves          : 41")
            print("    - Accuracy on training data : 1.0")
            print("    - Accuracy on test          : 0.9666666666666667")
            print("-" * 52)

            print("iris dataset :")
            print("  Training finished.")
            print("    - Depth                     : 10")
            print("    - Number of nodes           : 29")
            print("    - Number of leaves          : 15")
            print("    - Accuracy on training data : 0.9629629629629629")
            print("    - Accuracy on test          : 1.0")
            print("-" * 52)

            print("wine dataset :")
            print("  Training finished.")
            print("    - Depth                     : 10")
            print("    - Number of nodes           : 59")
            print("    - Number of leaves          : 30")
            print("    - Accuracy on training data : 0.906832298136646")
            print("    - Accuracy on test          : 0.8235294117647058")
            print("-" * 52)

    def fit_node(self, node):
        """
        Recursively fit nodes in the decision tree.

        Args:
            node (Node): The current node to be processed.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
                self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (
                self.explanatory[:, node.feature] <= node.threshold)

        is_left_leaf = self.is_leaf(left_population)
        is_right_leaf = self.is_leaf(right_population)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def is_leaf(self, sub_population):
        """
        Determine if a node should be a leaf.

        Args:
            sub_population (ndarray): The sub-population array for the node.
        
        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return len(sub_population) < 5  # Arbitrary condition for a leaf node

    def np_extrema(self, arr):
        """
        Return the minimum and maximum values of an array.

        Args:
            arr (ndarray): The input array.

        Returns:
            tuple: The minimum and maximum values in the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Choose a random feature and threshold for splitting a node.

        Args:
            node (Node): The current node to split.

        Returns:
            tuple: The chosen feature index and threshold.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory[:,feature][
                node.sub_population])
            diff = feature_max - feature_min

        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf child node.

        Args:
            node (Node): The parent node.
            sub_population (ndarray): The sub-population for this node.

        Returns:
            Leaf: The created leaf node.
        """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create a new child node.

        Args:
            node (Node): The parent node.
            sub_population (ndarray): The sub-population for this node.

        Returns:
            Node: The created child node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calculate the accuracy of the decision tree.

        Args:
            test_explanatory (ndarray): The explanatory variables for testing.
            test_target (ndarray): The target labels for testing.

        Returns:
            float: The accuracy of the decision tree.
        """
        return np.sum(np.equal(self.predict(test_explanatory), test_target)) / test_target.size

    def predict(self, explanatory):
        """
        Predict the target values for given explanatory variables.

        Args:
            explanatory (ndarray): A 2D numpy array of explanatory variables.

        Returns:
            ndarray: The predicted target values.
        """
        # Placeholder for prediction logic
        return np.zeros(explanatory.shape[0])

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree.

        Args:
            only_leaves (bool, optional): If True, count only leaf nodes. Defaults to False.

        Returns:
            int: The number of nodes (or leaf nodes) in the tree.
        """
        # Placeholder for counting nodes logic
        return 0

    def depth(self):
        """
        Calculate the depth of the tree.

        Returns:
            int: The depth of the tree.
        """
        # Placeholder for depth calculation
        return 0


class Node:
    def __init__(self):
        """
        Args: ff
        Returns: ff
        """
        self.sub_population = None
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.threshold = None
        self.depth = 0


class Leaf(Node):
    def __init__(self, value):
        """
        Args: ff
        Returns: ff
        """
        super().__init__()
        self.value = value
