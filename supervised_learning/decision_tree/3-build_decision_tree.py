#!/usr/bin/env python3
"""fonction"""

import numpy as np


class Node:
"""FFFFFFF"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Args: ff
        Returns: ff
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def get_leaves_below(self):
        """
        Args: ff
        Returns: ff
        """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


class Leaf(Node):
    def __init__(self, value, depth=None):
        """
        Args: ff
        Returns: ff
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Args: ff
        Returns: ff
        """
        return self.depth

    def get_leaves_below(self):
        """
        Args: ff
        Returns: ff
        """
        return [f"-> leaf [value={self.value}]"]


class Decision_Tree():
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """
        Args: ff
        Returns: ff
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Args: ff
        Returns: ff
        """
        return self.root.max_depth_below()

    def get_leaves(self):
        """
        Args: ff
        Returns: ff
        """
        return self.root.get_leaves_below()
