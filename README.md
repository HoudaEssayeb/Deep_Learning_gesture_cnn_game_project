# Gesture CNN Game Control — Sujet 21

Projet Python pour le **contrôle gestuel en temps réel d'un personnage de jeu vidéo par CNN**.\

Développement d'une application interactive capable de détecter les gestes d’un joueur à partir d’une caméra et de les convertir en commandes permettant de contrôler un personnage de jeu vidéo, notamment pour des actions telles que sauter, avancer, reculer ou se baisser.

Les CNN permettront de reconnaître les gestes à partir des images capturées, tandis qu’un module de traitement en temps réel assurera la transmission des commandes au jeu.

L’application proposera une interface permettant de visualiser les gestes reconnus, de tester les différentes commandes et d’évaluer la précision ainsi que le temps de réponse du système. 

Le projet utilise une webcam, un CNN MobileNetV2 (Transfer Learning), une interface Tkinter et une transmission de commandes clavier.

## Stack
- Python 3.15 recommandé
- PyTorch / torchvision : CNN, entraînement et inférence
- OpenCV : capture webcam et prétraitement
- Tkinter + Pillow : interface
- PyAutoGUI : commandes clavier
- scikit-learn / Matplotlib : évaluation

## Classes de gestes
Le dataset doit contenir les classes suivantes (noms de dossiers exacts) :
`idle`, `jump`, `forward`, `backward`, `crouch`

**Important :** vérifier que le dataset choisi contient ces poses ou préparer/annoter des images complémentaires. Un dataset ASL/alphabet ne correspond pas automatiquement à ces commandes.

## Installation (Windows / PowerShell)
```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
Installer PyTorch depuis le sélecteur officiel selon CPU/GPU : https://pytorch.org/get-started/locally/
Puis :
```powershell
pip install -r requirements.txt
```
Sélectionner l'interpréteur `.venv` dans VS Code.

## Dataset attendu
Les images doivent montrer la main/pose avec une définition de classe constante. Les splits doivent être distincts; éviter de placer des images quasi identiques dans train et test.

## Étapes
1. Collecter/préparer le dataset.
2. Entraîner : `python -m src.model.train`
3. Évaluer : `python -m src.model.evaluate`
4. Lancer l'application : `python main.py`

Le modèle entraîné est enregistré dans `models/best_model.pth`.

## Jeu vidéo
Le contrôleur envoie les touches (par défaut):
- `jump` → Space
- `right` → Right
- `left` → Left
- `roll` → Down
- `idle` → relâcher les touches

Tester d'abord dans un jeu qui accepte les commandes clavier. Le jeu/la fenêtre doit avoir le focus. Modifier le mapping dans `src/controller/game_controller.py` si nécessaire.

## Metriques
Mesure	Utilité
Accuracy	Pourcentage global de prédictions correctes
Precision	Fiabilité des prédictions pour chaque geste
Recall	Capacité à reconnaître chaque geste
F1-score	Équilibre entre precision et recall
Confusion matrix	Identification des erreurs entre les gestes

## Mesures affichées
- Geste prédit
- Confiance (softmax)
- FPS caméra / application
- Latence d'inférence (ms)

La latence affichée est le temps de l'inférence CNN, pas la latence complète webcam→jeu. Mesurer séparément la latence globale pour le rapport.

## Limites du prototype
- Les classes de gestes sont des poses statiques, classifiées frame par frame.
- Le seuil de confiance et la stabilité de prédiction sont des paramètres à calibrer.
- La qualité de reconnaissance dépend du dataset, de la lumière, de la distance et de la caméra.
