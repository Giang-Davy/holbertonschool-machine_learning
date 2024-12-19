#!/usr/bin/env python3

import numpy as np


class Node:
    # Classe représentant un nœud dans un arbre de décision
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        # Initialisation des bornes
        self.lower = {}  # Les bornes inférieures pour chaque feature
        self.upper = {}  # Les bornes supérieures pour chaque feature
        if feature is not None:
            self.lower[feature] = -100  # Valeur initiale arbitraire
            self.upper[feature] = 100   # Valeur initiale arbitraire

    def __str__(self):
        node_type = "root" if self.is_root else "node"
        details = f"{node_type} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            details += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            details += f"\n    +---> {right_str}"

        return details.rstrip()


class Leaf(Node):
    # Classe représentant une feuille dans l'arbre de décision
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"leaf [value={self.value}]"


class Decision_Tree:
    # Classe représentant un arbre de décision
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

    def __str__(self):
        return self.root.__str__()

    def update_predict(self):
        pass

    def update_bounds(self):
        # Mise à jour des bornes de l'arbre en fonction de ses noeuds
        def recursive_update(node):
            if node.is_leaf:
                return
            # Initialisation des bornes pour le feature du nœud
            if node.feature not in node.lower:
                node.lower[node.feature] = -100  # Valeur par défaut
            if node.feature not in node.upper:
                node.upper[node.feature] = 100  # Valeur par défaut
            
            node.lower[node.feature] = min(node.lower[node.feature], node.threshold)
            node.upper[node.feature] = max(node.upper[node.feature], node.threshold)
            recursive_update(node.left_child)
            recursive_update(node.right_child)

        recursive_update(self.root)

    def pred(self, sample):
        return self._predict_sample(sample, self.root)

    def _predict_sample(self, sample, node):
        if node.is_leaf:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left_child)
        else:
            return self._predict_sample(sample, node.right_child)

    def predict(self, X):
        predictions = [self._predict_sample(sample, self.root) for sample in X]
        return np.array(predictions)


def random_tree(max_depth, n_classes, n_features, seed=0):
    assert max_depth > 0, "max_depth must be a strictly positive integer"
    rng = np.random.default_rng(seed)
    root = Node(is_root=True, depth=0)
    root.lower = {i: -100 for i in range(n_features)}
    root.upper = {i: 100 for i in range(n_features)}

    def build_children(node):
        feat = rng.integers(0, n_features)
        node.feature = feat
        node.threshold = np.round(rng.uniform(0, 1) * (node.upper[feat] - node.lower[feat]) + node.lower[feat], 2)
        if node.depth == max_depth - 1:
            node.left_child = Leaf(depth=max_depth, value=rng.integers(0, n_classes))
            node.right_child = Leaf(depth=max_depth, value=rng.integers(0, n_classes))
        else:
            node.left_child = Node(depth=node.depth + 1)
            node.left_child.lower = node.lower.copy()
            node.left_child.upper = node.upper.copy()
            node.left_child.lower[feat] = node.threshold
            node.right_child = Node(depth=node.depth + 1)
            node.right_child.lower = node.lower.copy()
            node.right_child.upper = node.upper.copy()
            node.right_child.upper[feat] = node.threshold
            build_children(node.left_child)
            build_children(node.right_child)

    T = Decision_Tree(root=root)
    build_children(root)

    A = rng.uniform(0, 1, size=100 * n_features).reshape([100, n_features]) * 200 - 100
    return T, A


T, A = random_tree(4, 3, 5, seed=1)
print(T)

T.update_predict()
T.update_bounds()

print("T.pred(A) :\n", np.array([T.pred(x) for x in A]))
print("T.predict(A) :\n", T.predict(A))

test = np.all(np.equal(T.predict(A), np.array([T.pred(x) for x in A])))
print(f"Predictions are the same on the explanatory array A : {test}")
