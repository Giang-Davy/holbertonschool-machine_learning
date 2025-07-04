#!/usr/bin/env python3
"""play.py"""


import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from gymnasium.wrappers import AtariPreprocessing, FrameStack


# Wrapper pour compatibilité avec l'API Gym classique
class CompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def render(self, mode=None):
        # Ignore l'argument 'mode' pour la compatibilité
        return self.env.render()


# Création de l'environnement Atari Breakout
env = gym.make("ALE/Breakout-v5")
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                         frame_skip=1,
                         noop_max=30)
env = CompatibilityWrapper(env)

nb_actions = env.action_space.n
window_length = 4

# Définition du modèle CNN pour le DQN
model = Sequential()
# Permute pour passer de (window_length, 84, 84) à (84, 84, window_length)
model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Mémoire et politique d'exploration
memory = SequentialMemory(limit=100000, window_length=window_length)
policy = EpsGreedyQPolicy()

# Création et compilation de l'agent DQN
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    gamma=0.99,
    target_model_update=10000,
    delta_clip=1.0,
    policy=policy
)
dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

# Chargement des poids entraînés
dqn.load_weights("policy.h5")

# Lancement de 5 parties (sans affichage graphique ici)
dqn.test(env, nb_episodes=5, visualize=False)
