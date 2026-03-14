"""
predict.py
==========
Prédiction d'émotion sur une image + analyse enrichie
par l'API Google Gemini.

Auteur : Amine Annouka
Projet : FacialEmotionRecognition-CNN-Gemini
EST Sidi Bennour — Groupe 12 — 2025/2026

Utilisation
-----------
    python predict.py <chemin_image>
    python predict.py data/test_face.jpg
"""

import os
import sys
import json
import base64
import numpy as np
from pathlib import Path

import cv2
import tensorflow as tf
import google.generativeai as genai

from config import CONFIG

# ── Clé API Gemini ────────────────────────────────────────────────
# Définir la variable d'environnement GEMINI_API_KEY
# ou remplacer la chaîne ci-dessous par votre clé
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "VOTRE_CLE_GEMINI_ICI")
genai.configure(api_key=GEMINI_API_KEY)


# ── Chargement du modèle CNN ──────────────────────────────────────
def charger_modele() -> tf.keras.Model:
    """Charge le modèle CNN entraîné."""
    chemin = CONFIG["model_path"]
    if not Path(chemin).exists():
        print(f"\n❌  Modèle introuvable : {chemin}")
        print("   → Exécutez d'abord : python train.py\n")
        sys.exit(1)
    print(f"   ► Modèle chargé : {chemin}")
    return tf.keras.models.load_model(chemin)


# ── Prétraitement de l'image ──────────────────────────────────────
def pretraiter_image(chemin_image: str) -> np.ndarray:
    """
    Charge et prépare une image pour le réseau CNN.
    Conversion en niveaux de gris → redimensionnement 48×48
    → normalisation [0, 1].
    """
    img = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image illisible : {chemin_image}")
    img = cv2.resize(img, (48, 48))
    return (img.astype(np.float32) / 255.0).reshape(1, 48, 48, 1)


# ── Prédiction CNN ────────────────────────────────────────────────
def predire_emotion(modele: tf.keras.Model, chemin_image: str) -> dict:
    """
    Prédit l'émotion sur une image et retourne les probabilités.

    Retourne
    --------
    dict avec 'emotion', 'confiance', 'toutes_probabilites'
    """
    x       = pretraiter_image(chemin_image)
    probs   = modele.predict(x, verbose=0)[0]
    idx     = int(np.argmax(probs))
    return {
        "emotion"             : CONFIG["emotions"][idx],
        "emotion_fr"          : CONFIG["emotions_fr"][idx],
        "confiance"           : float(probs[idx]),
        "toutes_probabilites" : {
            emo: float(prob)
            for emo, prob in zip(CONFIG["emotions"], probs)
        },
    }


# ── Analyse Google Gemini ─────────────────────────────────────────
def analyser_avec_gemini(chemin_image: str, resultat_cnn: dict) -> str:
    """
    Envoie l'image et les résultats du CNN à Gemini Flash
    pour une analyse textuelle enrichie.
    """
    # Encodage base64
    with open(chemin_image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext_mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".bmp": "image/bmp",
        ".gif": "image/gif",
    }
    mime = ext_mime.get(Path(chemin_image).suffix.lower(), "image/jpeg")

    # Résumé des probabilités
    probs_str = "\n".join(
        f"  {e:<12} : {p * 100:5.1f}%"
        for e, p in sorted(
            resultat_cnn["toutes_probabilites"].items(),
            key=lambda x: x[1], reverse=True
        )
    )

    invite = f"""Tu es un expert en psychologie des émotions et en vision par ordinateur.

Un réseau de neurones convolutif a analysé ce visage avec les résultats suivants :
• Émotion principale : {resultat_cnn['emotion']} ({resultat_cnn['confiance'] * 100:.1f}% de confiance)
• Distribution complète :
{probs_str}

En observant l'image, réponds en français en 4 points numérotés :
1. Confirmes-tu l'émotion détectée ? Justifie ta réponse.
2. Quels indices visuels (yeux, sourcils, bouche, posture) observes-tu ?
3. Y a-t-il une ambiguïté possible, notamment entre Peur et Surprise ? Explique.
4. Donne un score de confiance de 0 à 10 pour la prédiction du réseau, avec justification.
"""

    modele_gemini = genai.GenerativeModel("gemini-1.5-flash")
    reponse = modele_gemini.generate_content([
        {"mime_type": mime, "data": b64},
        invite,
    ])
    return reponse.text


# ── Pipeline principal ────────────────────────────────────────────
def executer(chemin_image: str) -> None:
    """
    Pipeline complet : prédiction CNN + analyse Gemini + sauvegarde JSON.
    """
    print("\n" + "=" * 60)
    print(f"  Image : {chemin_image}")
    print("=" * 60)

    # 1. Prédiction CNN
    modele  = charger_modele()
    resultat = predire_emotion(modele, chemin_image)

    print(f"\n🧠  CNN → {resultat['emotion_fr']}  ({resultat['confiance'] * 100:.1f}%)\n")
    print("   Distribution complète :")

    for emo, prob in sorted(
        resultat["toutes_probabilites"].items(),
        key=lambda x: x[1], reverse=True
    ):
        barre = "█" * int(prob * 25)
        print(f"   {emo:<12} {barre:<26} {prob * 100:5.1f}%")

    # 2. Analyse Gemini
    print("\n🤖  Analyse Gemini en cours …\n")
    try:
        analyse_gemini = analyser_avec_gemini(chemin_image, resultat)
    except Exception as e:
        analyse_gemini = f"Erreur Gemini : {e}"
        print(f"   ⚠️  {analyse_gemini}")

    print("─" * 60)
    print(analyse_gemini)
    print("─" * 60)

    # 3. Sauvegarde JSON
    os.makedirs("results", exist_ok=True)
    sortie = {
        "image"           : chemin_image,
        "prediction_cnn"  : resultat,
        "analyse_gemini"  : analyse_gemini,
    }
    chemin_json = f"results/prediction_{Path(chemin_image).stem}.json"
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)
    print(f"\n💾  Résultats sauvegardés : {chemin_json}\n")


# ── Point d'entrée ────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage   : python predict.py <chemin_image>")
        print("Exemple : python predict.py data/test_face.jpg\n")
        sys.exit(1)
    executer(sys.argv[1])
