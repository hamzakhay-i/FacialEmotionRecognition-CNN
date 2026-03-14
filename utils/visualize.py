"""
utils/visualize.py
==================
Fonctions de visualisation : courbes d'apprentissage,
matrice de confusion, exemples d'images.

Auteur : Hamza Khayi
Projet : FacialEmotionRecognition-CNN-Gemini
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.makedirs("results", exist_ok=True)


# ── Exemples d'images par émotion ─────────────────────────────────
def plot_sample_images(
    X: np.ndarray,
    y: np.ndarray,
    emotions: list,
    n: int = 5,
    save_path: str = "results/sample_images.png"
) -> None:
    """Affiche n exemples pour chaque classe d'émotion."""
    y_labels = np.argmax(y, axis=1)
    fig, axes = plt.subplots(len(emotions), n, figsize=(n * 2, len(emotions) * 2))
    fig.suptitle("Exemples par classe d'émotion — FER2013",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, emo_idx in enumerate(range(len(emotions))):
        indices = np.where(y_labels == emo_idx)[0][:n]
        for col in range(n):
            ax = axes[row, col]
            if col < len(indices):
                ax.imshow(X[indices[col]].squeeze(), cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    emotions[emo_idx], fontsize=9,
                    rotation=0, labelpad=50, va="center"
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"   💾 {save_path}")
    plt.close()


# ── Courbes d'apprentissage ────────────────────────────────────────
def plot_history(
    history,
    save_path: str = "results/training_history.png"
) -> None:
    """Trace les courbes d'exactitude et de perte."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'apprentissage", fontsize=13, fontweight="bold")

    for ax, metric, title in [
        (ax1, "accuracy", "Exactitude (Accuracy)"),
        (ax2, "loss",     "Perte (Loss)"),
    ]:
        ax.plot(history.history[metric],
                label="Apprentissage", linewidth=2, color="#1565C0")
        ax.plot(history.history[f"val_{metric}"],
                label="Validation", linewidth=2, color="#E65100",
                linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Époque")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"   💾 {save_path}")
    plt.close()


# ── Matrice de confusion ───────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    emotions: list,
    save_path: str = "results/confusion_matrix.png"
) -> None:
    """
    Trace la matrice de confusion normalisée avec un zoom
    sur la confusion Peur / Surprise.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Analyse de la matrice de confusion",
                 fontsize=14, fontweight="bold")

    # ── Matrice complète ──────────────────────────────────────────
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=emotions, yticklabels=emotions,
        ax=axes[0], linewidths=0.4, vmin=0, vmax=1
    )
    axes[0].set_title("Matrice complète (normalisée)",
                      fontweight="bold", pad=12)
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")
    axes[0].tick_params(axis="x", rotation=45)

    # ── Zoom Peur / Surprise ──────────────────────────────────────
    fi = emotions.index("Fear")
    si = emotions.index("Surprise")
    sub = cm_norm[np.ix_([fi, si], [fi, si])]

    sns.heatmap(
        sub, annot=True, fmt=".2f", cmap="Reds",
        xticklabels=["Fear", "Surprise"],
        yticklabels=["Fear", "Surprise"],
        ax=axes[1], linewidths=1, annot_kws={"size": 18},
        vmin=0, vmax=1
    )
    axes[1].set_title(
        "Zoom : Peur ↔ Surprise (confusion principale)",
        fontweight="bold", color="darkred", pad=12
    )
    axes[1].set_xlabel("Prédit")
    axes[1].set_ylabel("Réel")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"   💾 {save_path}")
    plt.close()
