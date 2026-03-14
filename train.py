"""
train.py
========
Script d'entraînement principal du CNN pour la
reconnaissance d'expressions faciales sur FER2013.

Auteur : Mohamed Qennarouch
Projet : FacialEmotionRecognition-CNN-Gemini
EST Sidi Bennour — Groupe 12 — 2025/2026

Utilisation
-----------
    python train.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

from config import CONFIG
from utils.data_loader import load_fer2013
from utils.visualize   import (
    plot_history, plot_confusion_matrix, plot_sample_images,
)

# ── Reproductibilité ──────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
def build_cnn(num_classes: int = 7) -> tf.keras.Model:
    """
    Construit le réseau de neurones convolutif.

    Architecture
    ------------
    3 blocs convolutifs (Conv2D × 2 + BatchNorm + MaxPool + Dropout)
    + classifieur Dense (256 → 128 → 7)
    """
    model = Sequential([

        # ── Bloc 1 : détection des détails fins ──────────────────
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=(48, 48, 1), name="conv1_1"),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1_2"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Bloc 2 : détection des formes ────────────────────────
        Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_1"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_2"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Bloc 3 : détection des expressions ───────────────────
        Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3_1"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3_2"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Classifieur ───────────────────────────────────────────
        Flatten(),
        Dense(256, activation="relu", name="dense1"),
        BatchNormalization(),
        Dropout(0.50),
        Dense(128, activation="relu", name="dense2"),
        Dropout(0.30),
        Dense(num_classes, activation="softmax", name="output"),

    ], name="FER_CNN")

    model.compile(
        optimizer = Adam(learning_rate=CONFIG["learning_rate"]),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  RECONNAISSANCE D'EXPRESSIONS FACIALES — CNN")
    print("  EST Sidi Bennour — Groupe 12 — 2025/2026")
    print("=" * 60)

    # ── 1. Chargement des données ─────────────────────────────────
    print("\n📂 ÉTAPE 1 — Chargement du jeu de données FER2013")
    print("-" * 60)

    if not os.path.exists(CONFIG["data_path"]):
        print(f"\n❌  Fichier introuvable : {CONFIG['data_path']}")
        print("   → Téléchargez fer2013.csv sur :")
        print("     https://www.kaggle.com/datasets/msambare/fer2013")
        print("   → Placez-le dans le dossier data/\n")
        sys.exit(1)

    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(
        CONFIG["data_path"]
    )

    # Exemples visuels
    plot_sample_images(X_train, y_train, CONFIG["emotions"])

    # ── 2. Augmentation de données ────────────────────────────────
    print("\n🔄 ÉTAPE 2 — Augmentation de données")
    print("-" * 60)

    datagen = ImageDataGenerator(
        rotation_range     = 15,
        zoom_range         = 0.15,
        width_shift_range  = 0.10,
        height_shift_range = 0.10,
        horizontal_flip    = True,
        fill_mode          = "nearest",
    )
    datagen.fit(X_train)
    print("   Rotation ±15° | Zoom 15% | Shift 10% | Flip horizontal")

    # ── 3. Construction du modèle ─────────────────────────────────
    print("\n🧠 ÉTAPE 3 — Architecture du réseau")
    print("-" * 60)

    model = build_cnn()
    model.summary()

    # ── 4. Entraînement ───────────────────────────────────────────
    print("\n🚀 ÉTAPE 4 — Entraînement")
    print("-" * 60)
    print(f"   Époques max   : {CONFIG['epochs']}")
    print(f"   Taille lots   : {CONFIG['batch_size']}")
    print(f"   Taux initial  : {CONFIG['learning_rate']}\n")

    callbacks = [
        EarlyStopping(
            monitor             = "val_accuracy",
            patience            = CONFIG["early_stopping_patience"],
            restore_best_weights= True,
            verbose             = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = CONFIG["reduce_lr_factor"],
            patience = CONFIG["reduce_lr_patience"],
            min_lr   = CONFIG["min_lr"],
            verbose  = 1,
        ),
        ModelCheckpoint(
            filepath     = CONFIG["model_path"],
            monitor      = "val_accuracy",
            save_best_only = True,
            verbose      = 1,
        ),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=CONFIG["batch_size"]),
        validation_data = (X_val, y_val),
        epochs          = CONFIG["epochs"],
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── 5. Évaluation ─────────────────────────────────────────────
    print("\n📊 ÉTAPE 5 — Évaluation sur l'ensemble de test")
    print("-" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   ✅  Exactitude  : {test_acc * 100:.2f}%")
    print(f"   ✅  Perte       : {test_loss:.4f}")

    # Courbes d'apprentissage
    plot_history(history)

    # Matrice de confusion
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_true, y_pred, CONFIG["emotions"])

    # Rapport complet
    print("\n📋 Rapport de classification :\n")
    print(
        classification_report(
            y_true, y_pred,
            target_names=CONFIG["emotions"]
        )
    )

    # Analyse Peur / Surprise
    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(y_true, y_pred)
    fi = CONFIG["emotions"].index("Fear")
    si = CONFIG["emotions"].index("Surprise")
    print("🔍 Confusion Peur ↔ Surprise :")
    print(f"   Peur    → Surprise : {cm[fi][si]} cas")
    print(f"   Surprise→ Peur    : {cm[si][fi]} cas")

    print("\n" + "=" * 60)
    print(f"  ✅  Entraînement terminé !")
    print(f"  📁  Modèle sauvegardé : {CONFIG['model_path']}")
    print(f"  📁  Graphiques        : {CONFIG['results_dir']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
