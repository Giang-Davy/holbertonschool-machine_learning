#!/usr/bin/env python3
"""train.py"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Entraînement
    """

    # Historique des scores par épisode
    scores = []

    # Taille des états et nombre d'actions possibles dans l'environnement
    dimension_etats, nombre_actions = env.observation_space.shape[
        0], env.action_space.n

    # Initialisation aléatoire des paramètres (poids de la politique)
    poids = np.random.rand(dimension_etats, nombre_actions)

    for episode in range(nb_episodes):

        # Initialisation de l’environnement pour un nouvel épisode
        observation, _ = env.reset()
        episode_termine = False
        historique_recompenses = []
        historique_gradients = []

        while not episode_termine:
            # Choix de l’action et obtention du gradient associé
            action, gradient = policy_gradient(observation, poids)

            # Interaction avec l’environnement
            observation, recompense, episode_termine, interrompu, _ = env.step(
                action)

            # Stockage des gradients et des récompenses
            historique_gradients.append(gradient)
            historique_recompenses.append(recompense)

            # Mise à jour des poids avec la règle du gradient de politique
            for t, (g, r) in enumerate(zip(historique_gradients,
                                           historique_recompenses)):
                poids += alpha * g * (gamma ** t) * r

            if episode_termine:
                break

        env.close()
        scores.append(sum(historique_recompenses))

        # Affichage du score de l’épisode courant
        print(f"Episode: {episode} Score: {sum(historique_recompenses)}")
    return scores
