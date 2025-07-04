#!/usr/bin/env python3
"""train.py"""


import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from gymnasium.wrappers import AtariPreprocessing


# Wrapper pour compatibilité avec l'API Gym classique
class CompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        return obs


# Appuie automatiquement sur FIRE après chaque reset si nécessaire
class FireResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Si l'action 1 (FIRE) est valide, on la joue pour lancer la balle
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped,
                                                      'ale'):
            if self.env.unwrapped.ale.lives() == 5:
                obs, _, done, _ = self.env.step(1)
                if done:
                    obs = self.env.reset(**kwargs)
        return obs


def create_atari_environment(env_name):
    env = gym.make(env_name, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84,
                             grayscale_obs=True,
                             frame_skip=1, noop_max=30)
    env = CompatibilityWrapper(env)
    env = FireResetWrapper(env)  # Ajoute ce wrapper en dernier
    return env


def build_model(window_length, shape, actions):
    model = Sequential()
    # Permute pour adapter la forme des entrées au CNN
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


if __name__ == "__main__":
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n
    window_length = 4
    obs_shape = env.observation_space.shape  # (84, 84)
    model = build_model(window_length, obs_shape, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0,
        policy=policy
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    # Lance l'entraînement du DQN
    dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)
    # Sauvegarde les poids entraînés
    dqn.save_weights('policy.h5', overwrite=True)
    env.close()
