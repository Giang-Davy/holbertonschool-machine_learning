import numpy as np

class NeuralNetwork:
    """Réseau de neurones"""
    def __init__(self, nx, nodes):
        """Initialisation du réseau"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        
        # Initialisation des poids avec des valeurs très petites
        self._W1 = np.random.randn(nodes, nx) * 0.01  # Petits poids pour éviter des activations trop grandes
        self._b1 = np.ones((nodes, 1)) * 1  # Initialisation des biais à 1
        self._A1 = 0
        self._W2 = np.random.randn(1, nodes) * 0.01  # Petits poids pour éviter des activations trop grandes
        self._b2 = 1  # Initialisation du biais à 1
        self._A2 = 0

    @property
    def W1(self):
        """getter pour W1"""
        return self._W1

    @property
    def b1(self):
        """getter pour b1"""
        return self._b1

    @property
    def A1(self):
        """getter pour A1"""
        return self._A1

    @property
    def W2(self):
        """getter pour W2"""
        return self._W2

    @property
    def b2(self):
        """getter pour b2"""
        return self._b2

    @property
    def A2(self):
        """getter pour A2"""
        return self._A2

    def forward_prop(self, X):
        """Propagation avant"""
        Z1 = np.dot(self._W1, X) + self._b1
        self._A1 = 1 / (1 + np.exp(-Z1))  # Fonction sigmoïde
        
        Z2 = np.dot(self._W2, self._A1) + self._b2
        self._A2 = 1 / (1 + np.exp(-Z2))  # Fonction sigmoïde
        
        return self._A1, self._A2
