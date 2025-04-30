#!/usr/bin/env python3
"""2-wgan_gp.py"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """WGAN GP"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005, lambda_gp=10):
        """Initialization"""
        super().__init__()  # Run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3  # Standard value, but can be changed if necessary
        self.beta_2 = 0.9  # Standard value, but can be changed if necessary

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.reduce_mean(discriminator(x))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # Define the discriminator loss and optimizer:
        self.discriminator.loss = (
            lambda x, y: tf.reduce_mean(x) - tf.reduce_mean(y))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Generate real samples"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Generate interpolated samples"""
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute gradient penalty"""
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """Training step"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample(size=None)
                fake_sample = self.get_fake_sample(size=None, training=False)
                interpolated_sample = self.get_interpolated_sample(
                    real_sample, fake_sample)
                penalty = self.gradient_penalty(interpolated_sample)

                loss_real = self.discriminator(real_sample)
                loss_fake = self.discriminator(fake_sample)

                loss_discriminator = self.discriminator.loss(loss_real, loss_fake)
                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = loss_discriminator + self.lambda_gp * gp

            gradients = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            gen_loss = -tf.math.reduce_mean(self.discriminator(fake_sample))

        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": loss_discriminator, "gen_loss": gen_loss, "gp": penalty}
