# ================================================================
#  config.py — Configuration centrale du projet
#  FacialEmotionRecognition-CNN-Gemini
#  Auteurs : Hamza Khayi, Mohamed Qennarouch, Amine Annouka
#  EST Sidi Bennour — Groupe 12 — 2025/2026
# ================================================================

CONFIG = {
    # ── Données ──────────────────────────────────────────────────
    "data_path"   : "data/fer2013.csv",

    # ── Classes d'émotions (ordre officiel FER2013) ───────────────
    "emotions"    : [
        "Angry",     # 0 — Colère
        "Disgust",   # 1 — Dégoût
        "Fear",      # 2 — Peur
        "Happy",     # 3 — Joie
        "Sad",       # 4 — Tristesse
        "Surprise",  # 5 — Surprise
        "Neutral",   # 6 — Neutre
    ],

    # ── Traductions françaises ─────────────────────────────────────
    "emotions_fr" : [
        "Colère", "Dégoût", "Peur", "Joie",
        "Tristesse", "Surprise", "Neutre",
    ],

    # ── Image ─────────────────────────────────────────────────────
    "img_size"    : 48,
    "channels"    : 1,

    # ── Hyperparamètres ───────────────────────────────────────────
    "epochs"         : 80,
    "batch_size"     : 64,
    "learning_rate"  : 0.001,

    # ── Callbacks ─────────────────────────────────────────────────
    "early_stopping_patience"  : 15,
    "reduce_lr_patience"       : 5,
    "reduce_lr_factor"         : 0.5,
    "min_lr"                   : 1e-6,

    # ── Chemins ───────────────────────────────────────────────────
    "model_path"   : "models/best_model.keras",
    "results_dir"  : "results/",
}
