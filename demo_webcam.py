"""
demo_webcam.py
==============
Démonstration en temps réel de la reconnaissance d'expressions
faciales via la caméra, avec analyse Google Gemini à la demande.

Auteur : Amine Annouka
Projet : FacialEmotionRecognition-CNN-Gemini
EST Sidi Bennour — Groupe 12 — 2025/2026

Utilisation
-----------
    python demo_webcam.py

Commandes
---------
    A  →  Analyser le photogramme courant avec Google Gemini
    Q  →  Quitter l'application
"""

import os
import sys
import cv2
import base64
import threading
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from pathlib import Path

from config import CONFIG

# ── Clé API Gemini ────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "VOTRE_CLE_GEMINI_ICI")
genai.configure(api_key=GEMINI_API_KEY)

# ── Détecteur de visage (Haar Cascade) ───────────────────────────
DETECTEUR_VISAGE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Couleurs BGR par émotion ──────────────────────────────────────
COULEURS = {
    "Angry"   : (0,   0,   220),
    "Disgust" : (0,   128,   0),
    "Fear"    : (128,  0,  128),
    "Happy"   : (0,   215, 255),
    "Sad"     : (200, 100,   0),
    "Surprise": (0,   165, 255),
    "Neutral" : (150, 150, 150),
}

# ── Traductions affichées à l'écran ───────────────────────────────
NOMS_FR = {
    "Angry"   : "Colère",
    "Disgust" : "Dégoût",
    "Fear"    : "Peur",
    "Happy"   : "Joie",
    "Sad"     : "Tristesse",
    "Surprise": "Surprise",
    "Neutral" : "Neutre",
}


# ── Chargement du modèle ──────────────────────────────────────────
def charger_modele():
    chemin = CONFIG["model_path"]
    if not Path(chemin).exists():
        print(f"\n❌  Modèle introuvable : {chemin}")
        print("   → Exécutez d'abord : python train.py\n")
        sys.exit(1)
    print(f"⏳  Chargement du modèle CNN ({chemin}) …")
    modele = tf.keras.models.load_model(chemin)
    print("✅  Modèle chargé avec succès !\n")
    return modele


# ── Thread Gemini ─────────────────────────────────────────────────
texte_gemini  = ""
gemini_actif  = False
modele_gemini = genai.GenerativeModel("gemini-1.5-flash")


def interroger_gemini(frame_bgr: np.ndarray) -> None:
    """Analyse le photogramme avec Gemini (exécuté dans un thread séparé)."""
    global texte_gemini, gemini_actif
    gemini_actif = True
    try:
        _, buf = cv2.imencode(".jpg", frame_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        reponse = modele_gemini.generate_content([
            {"mime_type": "image/jpeg", "data": b64},
            (
                "Analyse l'expression faciale de la personne visible. "
                "Identifie l'émotion principale et cite 2 à 3 indices "
                "visuels observés (sourcils, yeux, bouche). "
                "Réponds en français en 3 phrases maximum."
            ),
        ])
        texte = reponse.text.strip().replace("\n", " ")
        # Découpe en lignes de 72 caractères pour l'affichage
        texte_gemini = "\n".join(
            texte[i: i + 72] for i in range(0, min(len(texte), 216), 72)
        )
    except Exception as e:
        texte_gemini = f"Erreur Gemini : {str(e)[:65]}"
    finally:
        gemini_actif = False


# ── Boucle principale ─────────────────────────────────────────────
def lancer_demo() -> None:
    modele = charger_modele()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Impossible d'accéder à la caméra.")
        sys.exit(1)

    print("🎥  Démonstration démarrée")
    print("   A → Analyser avec Gemini")
    print("   Q → Quitter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        visages = DETECTEUR_VISAGE.detectMultiScale(
            gris, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in visages:
            # ── Prédiction CNN ─────────────────────────────────────
            roi   = cv2.resize(gris[y: y + h, x: x + w], (48, 48))
            entree = roi.astype(np.float32).reshape(1, 48, 48, 1) / 255.0
            probs  = modele.predict(entree, verbose=0)[0]
            idx    = int(np.argmax(probs))
            emo    = CONFIG["emotions"][idx]
            conf   = probs[idx]
            coul   = COULEURS.get(emo, (255, 255, 255))
            nom_fr = NOMS_FR.get(emo, emo)

            # ── Rectangle + label principal ───────────────────────
            cv2.rectangle(frame, (x, y), (x + w, y + h), coul, 2)
            label = f"{nom_fr}  {conf * 100:.0f}%"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, coul, 2,
                        cv2.LINE_AA)

            # ── Barres de probabilités ─────────────────────────────
            bx = x + w + 10
            for i, (e, p) in enumerate(zip(CONFIG["emotions"], probs)):
                blen = int(p * 90)
                bc   = COULEURS.get(e, (180, 180, 180))
                cy   = y + i * 20
                cv2.rectangle(frame, (bx, cy), (bx + blen, cy + 15), bc, -1)
                cv2.putText(
                    frame,
                    f"{NOMS_FR.get(e, e)[:4]} {p * 100:.0f}%",
                    (bx + 95, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (230, 230, 230), 1, cv2.LINE_AA,
                )

        # ── Texte Gemini en bas ────────────────────────────────────
        if texte_gemini:
            for li, ligne in enumerate(texte_gemini.split("\n")):
                cv2.putText(
                    frame, ligne,
                    (10, frame.shape[0] - 65 + li * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 200), 1, cv2.LINE_AA,
                )

        # ── Barre de statut en haut ────────────────────────────────
        statut = (
            "Gemini : analyse en cours …"
            if gemini_actif
            else "A = Analyser avec Gemini   |   Q = Quitter"
        )
        cv2.putText(frame, statut, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(
            "FacialEmotionRecognition-CNN-Gemini  —  EST Sidi Bennour",
            frame,
        )

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        elif touche == ord("a") and not gemini_actif:
            threading.Thread(
                target=interroger_gemini,
                args=(frame.copy(),),
                daemon=True,
            ).start()

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋  Session terminée.\n")


# ── Point d'entrée ────────────────────────────────────────────────
if __name__ == "__main__":
    lancer_demo()
