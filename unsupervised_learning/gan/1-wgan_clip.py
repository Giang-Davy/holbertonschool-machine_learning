#!/usr/bin/env python3
"""1-wgan_clip.py"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples,
                 batch_size=200, disc_iter=2, learning_rate=0.005):
        """initialisation"""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Define optimizers
        self.generator.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.discriminator.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )

        # Define losses
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.discriminator.loss = (
            lambda x, y: tf.reduce_mean(y) - tf.reduce_mean(x))

    # Generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """avoir le faux"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # Generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """avoir le reel"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # Overloading train_step()
    def train_step(self, useless_argument):
        """Training step"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Generate real and fake samples
                real_sample = self.get_real_sample(size=None)
                fake_sample = self.get_fake_sample(size=None, training=False)

                # Compute discriminator outputs
                loss_real = self.discriminator(real_sample)
                loss_fake = self.discriminator(fake_sample)

                # Compute total discriminator loss
                loss_discriminator = tf.math.reduce_mean(loss_fake) - (
                    tf.math.reduce_mean(loss_real))

            # Compute discriminator gradients
            gradients = tape.gradient(
                loss_discriminator, self.discriminator.trainable_variables)

            # Apply gradients
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

            # Clip weights
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate a fake sample
            fake_sample = self.get_fake_sample(training=True)

            # Compute generator loss
            gen_loss = -tf.math.reduce_mean(self.discriminator(fake_sample))

        # Compute generator gradients
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)

        # Apply gradients
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        # Return losses
        return {"discr_loss": loss_discriminator, "gen_loss": gen_loss}
