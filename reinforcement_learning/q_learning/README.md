# Q-Learning sur FrozenLake

Ce projet implémente l'algorithme de Q-Learning pour résoudre l'environnement FrozenLake de [OpenAI Gymnasium](https://gymnasium.farama.org/).

## Structure du projet

- **0-load_env.py** : Charge l'environnement FrozenLake avec une carte personnalisée ou prédéfinie.
- **1-q_init.py** : Initialise la Q-table à zéro.
- **2-epsilon_greedy.py** : Implémente la stratégie d'exploration epsilon-greedy.
- **3-q_learning.py** : Entraîne l'agent avec l'algorithme Q-Learning.
- **4-play.py** : Fait jouer l'agent entraîné et affiche le rendu de l'environnement.
- **4-main.py** : Script principal pour entraîner et tester l'agent.
- **README.md** : Ce fichier.

## Installation

1. Installez les dépendances nécessaires :
   ```sh
   pip install gymnasium numpy
   ```
