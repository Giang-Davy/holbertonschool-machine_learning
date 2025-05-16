#!/usr/bin/env python3
"""forecast_btc.py"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt  # Importer matplotlib pour tracer les courbes


def main():
    # Charger une partie des données pour tester
    df = pd.read_csv("./coinbase.csv").head(10000)
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
                        'Volume_(Currency)', 'Weighted_Price']

    # Supprimer les valeurs manquantes ou infinies
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Normaliser les données
    scaler = MinMaxScaler()
    df[required_columns] = scaler.fit_transform(df[required_columns])

    # Générer les séquences X et y
    sequence = 144
    X, y = [], []
    for i in range(sequence, len(df) - 60):
        X.append(df[required_columns].iloc[i - sequence:i].values)
        y.append(df['Close'].iloc[i + 60])

    X = np.array(X)
    y = np.array(y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Diviser les données en ensembles d'entraînement et de validation
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Construire un modèle simplifié
    optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, input_shape=(144, 7),
                             kernel_regularizer=tf.keras.regularizers.l2(
                                 0.01)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss="mse", metrics=['mae'])

    # Entraîner le modèle
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=15, verbose=1)

    # Sauvegarder le modèle
    model.save("time_series.keras")

    # Évaluer le modèle
    score = model.evaluate(X_val, y_val)
    print("Loss:", score[0])
    print("Mean Absolute Error:", score[1])

    # Tracer les courbes de perte
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title('Courbe de perte')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
