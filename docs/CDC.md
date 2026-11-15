# Cahier des charges — Sujet 21

## Objectif
Développer une application interactive qui classe des gestes de la main capturés par webcam à l'aide d'un CNN et les convertit en commandes de contrôle d'un personnage de jeu vidéo.

## Fonctionnalités
- Capture vidéo webcam.
- Prétraitement d'image.
- Classification CNN de cinq classes (idle, jump, forward, backward, crouch).
- Transmission clavier au jeu.
- GUI : flux vidéo, classe prédite, confiance, FPS, latence d'inférence.
- Évaluation sur dataset de test : accuracy, precision/recall/F1, matrice de confusion.
- Mesure du temps de réponse.

## Contraintes
- CNN doit être réellement entraîné/fine-tuné et utilisé en inférence.
- Le dataset doit correspondre aux gestes choisis.
- Les jeux de données train/validation/test doivent être distincts.
- La latence et le FPS sont mesurés sur l'environnement de test et reportés dans le rapport.

## Metrics
Mesure	Utilité
Accuracy	Pourcentage global de prédictions correctes
Precision	Fiabilité des prédictions pour chaque geste
Recall	Capacité à reconnaître chaque geste
F1-score	Équilibre entre precision et recall
Confusion matrix	Identification des erreurs entre les gestes

## Requirements
Bibliothèque	Utilisation
torch	CNN et Deep Learning
torchvision	MobileNetV2 et transformations d'images
opencv-python	Webcam et traitement vidéo
numpy	Manipulation des données
Pillow	Traitement des images
scikit-learn	Métriques d'évaluation
matplotlib	Courbes et graphiques
customtkinter	Interface graphique
pyautogui	Simulation des commandes clavier
tqdm	Progression de l'entraînement