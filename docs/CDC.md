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
