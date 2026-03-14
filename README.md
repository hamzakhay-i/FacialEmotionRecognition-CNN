# 🎭 FacialEmotionRecognition-CNN-Gemini

> Reconnaissance d'expressions faciales en temps réel : CNN entraîné sur FER2013 (7 émotions) enrichi par l'API Google Gemini pour une analyse textuelle intelligente.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4?logo=google)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Description

Ce projet implémente un système complet de **reconnaissance d'expressions faciales** combinant :
- Un **Réseau de Neurones Convolutif (CNN)** entraîné sur le jeu de données **FER2013**
- L'**API Google Gemini** pour une analyse textuelle enrichie des émotions détectées
- Une **application webcam en temps réel** avec barres de probabilités

**7 émotions reconnues :** 😠 Colère · 🤢 Dégoût · 😨 Peur · 😄 Joie · 😢 Tristesse · 😲 Surprise · 😐 Neutre

---

## ✨ Fonctionnalités

- ✅ Classification de 7 émotions avec ~66% d'exactitude
- ✅ Augmentation de données (rotation, zoom, miroir)
- ✅ Analyse Gemini : confirmation, indices visuels, score de confiance
- ✅ Démonstration webcam temps réel avec détection de visage
- ✅ Matrice de confusion + courbes d'apprentissage
- ✅ Sauvegarde automatique du meilleur modèle

---

## 🗂️ Structure du projet

```
FacialEmotionRecognition-CNN-Gemini/
│
├── train.py                  # Entraînement du CNN
├── predict.py                # Prédiction image + analyse Gemini
├── demo_webcam.py            # Démonstration temps réel
├── config.py                 # Configuration centrale
├── requirements.txt          # Dépendances Python
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Chargement et prétraitement FER2013
│   └── visualize.py          # Graphiques et matrices
│
├── data/
│   └── fer2013.csv           # À télécharger sur Kaggle
│
├── models/
│   └── best_model.keras      # Généré après entraînement
│
└── results/                  # Graphiques générés
    ├── training_history.png
    ├── confusion_matrix.png
    └── sample_images.png
```

---

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/FacialEmotionRecognition-CNN-Gemini.git
cd FacialEmotionRecognition-CNN-Gemini
```

### 2. Créer l'environnement (recommandé : Anaconda)

```bash
conda create -n fer python=3.11 -y
conda activate fer
pip install -r requirements.txt
```

### 3. Télécharger FER2013

Télécharger `fer2013.csv` sur [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) et le placer dans `data/`

### 4. Configurer la clé API Gemini (gratuite)

Créer une clé sur [Google AI Studio](https://aistudio.google.com/app/apikey) et la définir :

```bash
# Windows
set GEMINI_API_KEY=votre_cle_ici

# macOS / Linux
export GEMINI_API_KEY=votre_cle_ici
```

---

## 🚀 Utilisation

### Entraîner le modèle

```bash
python train.py
```

### Prédire sur une image

```bash
python predict.py data/test_face.jpg
```

### Lancer la démonstration webcam

```bash
python demo_webcam.py
# Touches : A = Analyser avec Gemini | Q = Quitter
```

---

## 🧠 Architecture CNN

```
Input (48×48×1)
    │
    ├── Bloc 1 : Conv2D(32) × 2 + BatchNorm + MaxPool + Dropout(0.25)
    ├── Bloc 2 : Conv2D(64) × 2 + BatchNorm + MaxPool + Dropout(0.25)
    ├── Bloc 3 : Conv2D(128) × 2 + BatchNorm + MaxPool + Dropout(0.25)
    │
    ├── Flatten → Dense(256) + BatchNorm + Dropout(0.50)
    ├── Dense(128) + Dropout(0.30)
    └── Dense(7, softmax) → 7 émotions
```

---

## 📊 Résultats

| Émotion   | Exactitude |
|-----------|-----------|
| 😄 Joie   | ~88%       |
| 😲 Surprise | ~75%     |
| 😐 Neutre | ~74%       |
| 😢 Tristesse | ~68%    |
| 😠 Colère | ~62%       |
| 😨 Peur   | ~48%       |
| 🤢 Dégoût | ~38%       |
| **Global**| **~66%**   |

---

## 👥 Auteurs

**Groupe 12 — École Supérieure de Technologie, Sidi Bennour (2025-2026)**

- **Hamza Khayi** — Chargement des données & Prétraitement
- **Mohamed Qennarouch** — Architecture CNN & Entraînement
- **Amine Annouka** — API Gemini & Démonstration webcam

---

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
