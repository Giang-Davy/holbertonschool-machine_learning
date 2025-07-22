#!/usr/bin/env python3
"""
Script d'entraînement d'un agent DQN pour Breakout (version modifiée)

Ce fichier implémente une approche DQN avec prétraitement visuel et un
réseau convolutif. Il est adapté pour l'utilisation avec keras-rl2 et
assure la compatibilité avec les environnements Gymnasium.
"""

import numpy as np
from PIL import Image
import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


class CompatEnv:
    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs, _ = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated or truncated, info

    def render(self, *args, **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()


class FrameProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        image = Image.fromarray(observation)
        image = image.resize((84, 84)).convert('L')
        return np.array(image).astype('uint8')

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def create_network(action_size):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(action_size))
    model.add(Activation('linear'))
    return model


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    env = CompatEnv(env)
    actions = env.action_space.n
    model = create_network(actions)
    model.summary()

    memory = SequentialMemory(limit=1000000, window_length=4)
    processor = FrameProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1.0, value_min=0.05, value_test=0.01,
                                  nb_steps=50000)

    dqn = DQNAgent(model=model,
                   nb_actions=actions,
                   memory=memory,
                   processor=processor,
                   policy=policy,
                   nb_steps_warmup=10000,
                   gamma=0.99,
                   target_model_update=1000,
                   train_interval=4,
                   delta_clip=1.0)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    print("Training begins...")
    dqn.fit(env, nb_steps=100000, visualize=False, verbose=2, nb_max_episode_steps=10000)
    dqn.save_weights('policy.h5', overwrite=True)
