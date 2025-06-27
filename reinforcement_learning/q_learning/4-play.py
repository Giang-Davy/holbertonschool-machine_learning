#!/usr/bin/env python3
"""4-play.py"""


import numpy as np


def play(env, Q, max_steps=100):
    """agent entrainé qui joue un episode"""
    rewards = 0
    # Réinitialisation de l’environnement
    state = env.reset()
    render_list = []
    # Garantire la compatibilité avec les versions de gymnasium
    if isinstance(state, tuple):
        state = state[0]
    for step in range(max_steps):
        render = env.render()
        # Sauvegarder le rendu
        render_list.append(render)
        # Exectuter une action
        best_action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(
                best_action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
            # Rassembler toute les récompenses
        rewards += reward
        state = next_state  # mettre à jour l'état
        done = terminated or truncated
        # Si l'épisode est finie, on sort de la boucle
        if done:
            break

    return rewards, render_list
