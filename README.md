
# Gesture CNN Game Control — Sujet 21

Application Python de contrôle gestuel en temps réel d'un personnage de jeu vidéo à l'aide d'un réseau de neurones convolutifs (CNN).

Le système utilise une webcam pour détecter cinq gestes de la main, les classifier avec MobileNetV2 et les convertir en commandes clavier.

## 1. Objectifs

- Reconnaître les gestes de la main à partir d'une webcam.
- Utiliser MobileNetV2 avec Transfer Learning.
- Stabiliser les prédictions successives.
- Contrôler un jeu vidéo avec les commandes clavier.
- Visualiser les prédictions et les performances du modèle.

## 2. Technologies

- Python
- PyTorch et torchvision
- MobileNetV2
- OpenCV
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- CustomTkinter
- PyAutoGUI
- tqdm

## 3. Classes de gestes

| Classe | Geste | Commande |
|---|---|---|
| idle | Main ouverte | Aucune |
| jump | Pouce vers le haut | Up |
| left | Pouce orienté à gauche | Left |
| right | Pouce orienté à droite | Right |
| roll | Pouce vers le bas | Down |

Le jeu doit accepter les commandes clavier et posséder le focus clavier.

## 4. Installation et exécution

### 4.1 Création de l'environnement

Sous Windows / PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Installer PyTorch et torchvision en fonction de l'environnement CPU/GPU depuis :

https://pytorch.org/get-started/locally/

Installer ensuite les dépendances :

```powershell
pip install -r requirements.txt
```

Sélectionner l'interpréteur `.venv` dans VS Code.

### 4.2 Préparation du dataset

Les scripts disponibles permettent de constituer, vérifier et préparer les données.

```powershell
python collect_dataset.py
python check_dataset.py
python rename_dataset_images.py
python prepare_dataset.py
```

La préparation organise les images dans les ensembles :

- `train`
- `val`
- `test`

### 4.3 Entraînement du modèle

```powershell
python train.py
```

Le modèle sélectionné sur la base des performances de validation est sauvegardé dans :

```text
models/mobilenetv2_best.pth
```

L'historique d'entraînement est enregistré dans :

```text
models/training_history.csv
```

### 4.4 Évaluation du modèle

```powershell
python evaluate.py
```

L'évaluation génère notamment :

- Accuracy globale.
- Precision, recall et F1-score par classe.
- Matrice de confusion.
- Rapport d'évaluation.

Les résultats sont enregistrés dans le dossier `models/`.

### 4.5 Lancement de l'application

```powershell
python main.py
```

L'application permet de démarrer la webcam, visualiser les gestes reconnus, activer le contrôle clavier et consulter le tableau de bord des performances.

### 4.6 Exécution des tests

Depuis la racine du projet :

```powershell
python -m tests.test_camera
python -m tests.test_model
python -m tests.test_preprocessing
python -m tests.test_stabilizer
python -m tests.test_game_controller
python -m tests.test_game_keys
```

## 5. Mesures et évaluation

Les principales métriques utilisées sont :

- Accuracy : proportion de prédictions correctes.
- Precision : fiabilité des prédictions par classe.
- Recall : capacité à retrouver les exemples d'une classe.
- F1-score : équilibre entre precision et recall.
- Confusion matrix : analyse des erreurs de classification.

L'application affiche notamment le geste reconnu, la confiance et le temps d'inférence.

Le temps d'inférence ne représente pas la latence complète entre la présentation du geste et la réaction du jeu.

## 6. Structure du projet

```text
DEEP_LEARNING_GESTURE_CNN_GAME_PROJECT/
│
├── main.py
├── train.py
├── evaluate.py
├── collect_dataset.py
├── check_dataset.py
├── prepare_dataset.py
├── rename_dataset_images.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── camera/
│   ├── controller/
│   ├── gui/
│   └── model/
│
├── tests/
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── models/
└── docs/
```

## 7. Limites

- Les gestes sont des poses statiques classifiées image par image.
- Les performances dépendent de la qualité et de la diversité du dataset.
- La lumière, la distance et la position de la main peuvent influencer la reconnaissance.
- Le seuil de confiance et le stabilisateur doivent être calibrés.
- Le jeu doit accepter les touches clavier envoyées par l'application.
