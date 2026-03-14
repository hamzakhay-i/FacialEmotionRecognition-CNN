"""
utils/data_loader.py
====================
Chargement et prétraitement du jeu de données FER2013.

Auteur : Hamza Khayi
Projet : FacialEmotionRecognition-CNN-Gemini
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_fer2013(csv_path: str):
    """
    Lit le fichier fer2013.csv et retourne les ensembles
    d'apprentissage, de validation et de test.

    Paramètres
    ----------
    csv_path : str
        Chemin vers le fichier fer2013.csv

    Retourne
    --------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Tableaux normalisés (float32) et étiquettes encodées.
    """
    print(f"\n   ► Lecture de : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ► {len(df):,} images chargées.")

    # ── 1. Conversion des pixels ──────────────────────────────────
    def parse_pixels(pixel_string: str) -> np.ndarray:
        """Convertit une chaîne de pixels en tenseur 48×48×1 normalisé."""
        pixels = np.array(pixel_string.split(), dtype=np.float32)
        return (pixels.reshape(48, 48, 1) / 255.0)

    X = np.array([parse_pixels(p) for p in df["pixels"]], dtype=np.float32)

    # ── 2. Encodage des étiquettes ────────────────────────────────
    y = to_categorical(df["emotion"].values, num_classes=7)

    # ── 3. Découpage des données ──────────────────────────────────
    if "Usage" in df.columns:
        # Découpage officiel FER2013
        mask_train = df["Usage"] == "Training"
        mask_val   = df["Usage"] == "PublicTest"
        mask_test  = df["Usage"] == "PrivateTest"

        X_train, y_train = X[mask_train], y[mask_train]
        X_val,   y_val   = X[mask_val],   y[mask_val]
        X_test,  y_test  = X[mask_test],  y[mask_test]
    else:
        # Découpage automatique 80/10/10
        labels = np.argmax(y, axis=1)
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=labels
        )
        labels_tmp = np.argmax(y_tmp, axis=1)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=labels_tmp
        )

    print(f"   ► Apprentissage : {len(X_train):>6,} images")
    print(f"   ► Validation    : {len(X_val):>6,} images")
    print(f"   ► Test          : {len(X_test):>6,} images")

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Calcule les poids de classes pour gérer le déséquilibre du dataset.

    Paramètres
    ----------
    y_train : np.ndarray
        Étiquettes d'apprentissage encodées (one-hot).

    Retourne
    --------
    dict : {classe: poids}
    """
    labels   = np.argmax(y_train, axis=1)
    n_total  = len(labels)
    n_classes = 7
    weights = {}

    for c in range(n_classes):
        n_c = np.sum(labels == c)
        weights[c] = n_total / (n_classes * n_c) if n_c > 0 else 1.0

    return weights
