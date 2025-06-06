#!/usr/bin/env python3
"""5-train.py"""


import numpy as np
import tensorflow as tf


Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """entrainement avec transformer"""
    dataset = Dataset(batch_size, max_len)
    input_vocab_size = len(dataset.tokenizer_pt.get_vocab()) + 2
    target_vocab_size = len(dataset.tokenizer_en.get_vocab()) + 2

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super().__init__()
            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * tf.math.pow(self.warmup_steps, -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss = loss_object(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    model = Transformer(N, dm, h, hidden,
                        input_vocab_size, target_vocab_size,
                        max_len, max_len)

    for epoch in range(epochs):
        batch = 0
        for inputs, targets in dataset.data_train:
            tar_inp = targets[:, :-1]
            tar_real = targets[:, 1:]

            encoder_mask, look_ahead_mask, decoder_mask = create_masks(inputs, tar_inp)

            with tf.GradientTape() as tape:
                predictions = model(inputs, tar_inp, training=True,
                                    encoder_mask=encoder_mask,
                                    look_ahead_mask=look_ahead_mask,
                                    decoder_mask=decoder_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(tar_real, predictions)

            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}, batch {batch}: loss {train_loss.result():.4f} accuracy {train_accuracy.result():.4f}")
            batch += 1

        print(f"Epoch {epoch + 1} summary: loss {train_loss.result():.4f} accuracy {train_accuracy.result():.4f}")
        train_loss.reset_states()
        train_accuracy.reset_states()

    return model
