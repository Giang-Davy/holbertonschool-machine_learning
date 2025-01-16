#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """entrainement avec early stopping"""
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=verbose,
            restore_best_weights=True
        )
        history = network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_data,
            shuffle=shuffle,
            callbacks=[early_stopping_callback]
        )
    else:
        history = network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_data,
            shuffle=shuffle
        )

    return history
