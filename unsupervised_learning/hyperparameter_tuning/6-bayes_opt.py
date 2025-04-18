#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import GPyOpt
from joblib import dump
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Chargement des données
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Liste pour stocker les valeurs de la fonction objectif
objective_values = []

# Fonction objective avec TensorFlow
def objective_function(X):
    X = X[0]
    n_units = int(X[0])  # Nombre de neurones dans la couche cachée
    learning_rate = X[1]  # Taux d'apprentissage
    batch_size = int(X[2])  # Taille des lots
    epochs = int(X[3])  # Nombre d'époques

    # Définir le modèle
    model = Sequential([
        Dense(n_units, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')  # Sortie binaire
    ])

    # Compiler le modèle
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Définir le chemin du checkpoint
    checkpoint_path = f"checkpoint_nunits{n_units}_lr{learning_rate:.4f}_batch{batch_size}_epochs{epochs}.h5"

    # Callback pour sauvegarder le meilleur modèle
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',  # Surveiller la précision sur les données de validation
        save_best_only=True,
        verbose=0
    )

    # Entraîner le modèle
    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,  # Utiliser une partie des données d'entraînement pour la validation
        verbose=0,
        callbacks=[checkpoint_callback]
    )

    # Évaluer le modèle
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Hyperparameters: {X}, Accuracy: {accuracy}")

    # Stocker la précision négative pour le graphique final
    objective_values.append(-accuracy)

    return -accuracy  # Retourne la précision négative pour minimisation

# Définition des domaines des hyperparamètres
domain = [
    {'name': 'n_units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},  # Nombre de neurones
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},  # Taux d'apprentissage
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)},  # Taille des lots
    {'name': 'epochs', 'type': 'discrete', 'domain': (10, 20, 30)}  # Nombre d'époques
]

# Optimisation bayésienne avec GPyOpt
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=domain,
    acquisition_type='EI',
    exact_eval=True,
    initial_design_numdata=10
)

# Lancer l'optimisation
optimizer.run_optimization(max_iter=30)

# Affichage des résultats de l'optimisation
print("Meilleure configuration :")
print(f"n_units = {optimizer.x_opt[0]}")
print(f"learning_rate = {optimizer.x_opt[1]}")
print(f"batch_size = {optimizer.x_opt[2]}")
print(f"epochs = {optimizer.x_opt[3]}")

# Écriture des résultats dans un fichier .txt
with open("optimization_results.txt", "w") as file:
    file.write("Meilleure configuration :\n")
    file.write(f"n_units = {optimizer.x_opt[0]}\n")
    file.write(f"learning_rate = {optimizer.x_opt[1]}\n")
    file.write(f"batch_size = {optimizer.x_opt[2]}\n")
    file.write(f"epochs = {optimizer.x_opt[3]}\n")
    file.write("\nValeurs de la fonction objectif :\n")
    for i, value in enumerate(objective_values):
        file.write(f"Itération {i + 1}: {value}\n")

# Tracer le graphique de convergence à la fin
plt.plot(range(len(objective_values)), objective_values, marker='o')
plt.xlabel('Itérations')
plt.ylabel('Valeur de la fonction objectif (précision négative)')
plt.title('Convergence de l\'optimisation bayésienne')
plt.show()
