#!/usr/bin/env python3
"""preprocess_data.py"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Chargement des fichiers CSV
df_coinbase = pd.read_csv("./coinbase.csv")
df_bitstamp = pd.read_csv("./bitstamp.csv")

# Vérification des colonnes
print("Colonnes dans coinbase.csv :", df_coinbase.columns)
print("Colonnes dans bitstamp.csv :", df_bitstamp.columns)

if not df_coinbase.columns.equals(df_bitstamp.columns):
    raise ValueError("Les colonnes des fichiers CSV ne sont pas les mêmes")

# Vérification de la présence des colonnes nécessaires
required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close',
                    'Volume_(BTC)', 'Volume_(Currency)',
                    'Weighted_Price']
for col in required_columns:
    if col not in df_coinbase.columns:
        raise KeyError(
            f"Colonne '{col}' absente dans coinbase.csv.{df_coinbase.columns}")
    if col not in df_bitstamp.columns:
        raise KeyError(
            f"Colonne '{col}' absente dans bitstamp.csv.{df_bitstamp.columns}")

# Prétraitement des données
df = pd.concat([df_coinbase, df_bitstamp], ignore_index=True)
df = df.sort_values(by='Timestamp')
df = df.dropna()
df = df.drop_duplicates(subset='Timestamp')
df = df.reset_index(drop=True)
df = df[required_columns]

# Normalisation des données
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

# Recréation du DataFrame avec les colonnes d'origine
df_normalized = pd.DataFrame(df_normalized, columns=df.columns)

# Affichage des premières lignes du DataFrame normalisé
print(df_normalized.head())
df_normalized.to_csv("preprocessed_data.csv", index=False)
