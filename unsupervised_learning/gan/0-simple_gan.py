#!/usr/bin/env python3
"""0-simple_gan.py"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """initialisation"""
        super().__init__()  # Run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5  # Standard value, but can be changed if necessary
        self.beta_2 = 0.9  # Standard value, but can be changed if necessary

        # Define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # Define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # Generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """faux sample"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # Generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """vrai sample"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # Overloading train_step()
    def train_step(self, useless_argument):
        """entrainement"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Generate a real sample and a fake sample
                real_sample = self.get_real_sample(size=None)
                fake_sample = self.get_fake_sample(size=None, training=False)

                # Compute discriminator output for real and fake samples
                loss_real = self.discriminator(real_sample)
                loss_fake = self.discriminator(fake_sample)

                # Compute total discriminator loss
                loss_diff = (
                    tf.keras.losses.MeanSquaredError()(
                        tf.ones_like(loss_real), loss_real) +
                    tf.keras.losses.MeanSquaredError()(
                        -1 * tf.ones_like(loss_fake), loss_fake))

            # Compute discriminator gradients
            gradients = tape.gradient(loss_diff,
                                      self.discriminator.trainable_variables)

            # Apply gradients
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate a fake sample
            fake_sample = self.get_fake_sample(training=True)

            # Compute generator loss
            gen_loss = tf.keras.losses.MeanSquaredError()(
                tf.ones_like(self.discriminator(fake_sample)),
                self.discriminator(fake_sample))

        # Compute generator gradients
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)

        # Apply gradients
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))

        # Return losses
        return {"discr_loss": loss_diff, "gen_loss": gen_loss}
