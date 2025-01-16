#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """entrainement avec early stopping et décroissance"""

    callbacks = []

    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=verbose,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_callback)

    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_scheduler = K.callbacks.LearningRateScheduler(
                                   lr_schedule, verbose=1)
        callbacks.append(lr_scheduler)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
