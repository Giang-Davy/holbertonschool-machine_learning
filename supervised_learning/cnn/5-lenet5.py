#!/usr/bin/env python3
"""LeNet-5 modifié"""

from tensorflow import keras as K

def lenet5(X):
    """modèle LeNet-5"""
    initializer = K.initializers.VarianceScaling(scale=2.0, seed=0)
    
    # Première couche convolutionnelle
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation='relu', kernel_initializer=initializer)(X)
    # Première couche de pooling
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    
    # Deuxième couche convolutionnelle
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation='relu', kernel_initializer=initializer)(pool1)
    # Deuxième couche de pooling
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    
    # Aplatissement des données
    flat = K.layers.Flatten()(pool2)
    
    # Première couche entièrement connectée
    fc1 = K.layers.Dense(units=120, activation='relu',
                          kernel_initializer=initializer)(flat)
    
    # Deuxième couche entièrement connectée
    fc2 = K.layers.Dense(units=84, activation='relu',
                          kernel_initializer=initializer)(fc1)
    
    # Couche de sortie softmax
    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=initializer)(fc2)
    
    # Création du modèle
    model = K.Model(inputs=X, outputs=output)
    
    # Compilation du modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
