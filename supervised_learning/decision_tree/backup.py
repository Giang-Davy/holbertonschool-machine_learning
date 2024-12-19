#!/usr/bin/env python3
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        if self.is_leaf:
            return self.depth
        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left_depth, right_depth)

    def __str__(self):
        result = f"root [feature={self.feature}, threshold={self.threshold}]" if self.is_root else f"node [feature={self.feature}, threshold={self.threshold}]"
        if self.left_child:
            result += "\n" + left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            result += "\n" + right_child_add_prefix(self.right_child.__str__())
        return result

    def pred(self, sample):
        """
        Prédit la classe pour un seul exemple (un vecteur).
        """
        if self.is_leaf:
            return self.value
        if sample[self.feature] <= self.threshold:
            return self.left_child.pred(sample) if self.left_child else self.value
        else:
            return self.right_child.pred(sample) if self.right_child else self.value

def left_child_add_prefix(text):
    lines = text.split("\n")
    new_text = "    +---> " + lines[0]
    for x in lines[1:]:
        new_text += "\n    |  " + x
    return new_text

def right_child_add_prefix(text):
    lines = text.split("\n")
    new_text = "    +---> " + lines[0]
    for x in lines[1:]:
        new_text += "\n       " + x
    return new_text


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def __str__(self):
        return f"-> leaf [value={self.value}]"

class Decision_Tree():
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()

    def __str__(self):
        return self.root.__str__()

    def print_tree(self):
        tree_str = self.root.__str__()
        tree_lines = tree_str.splitlines()
        result = []
        for line in tree_lines:
            result.append(line)
            if line.strip().startswith("root") or line.strip().startswith("node"):
                result.append("")  # Ajouter un saut de ligne
        return "\n".join(result).strip()

    def pred(self, sample):
        """
        Utilise la méthode 'pred' du noeud pour faire une prédiction sur un exemple (ligne du tableau).
        """
        return self.root.pred(sample)

    def update_predict(self):
        """
        Met à jour la fonction de prédiction pour l'ensemble des exemples.
        """
        def predict_fn(data):
            return np.array([self.pred(x) for x in data])
        self.predict = predict_fn
