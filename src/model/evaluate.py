
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.model.cnn_model import build_model, CLASS_NAMES


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "mobilenetv2_best.pth"

REPORT_PATH = MODEL_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.csv"

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Chargement du dataset de test
# --------------------------------------------------

def load_test_dataset():

    weights = models.MobileNet_V2_Weights.DEFAULT

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=weights.transforms()
    )

    expected_classes = list(CLASS_NAMES)

    if test_dataset.classes != expected_classes:
        raise ValueError(
            f"Incorrect classes: {test_dataset.classes}. "
            f"Expected: {expected_classes}"
        )

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return test_loader


# --------------------------------------------------
# 3. Chargement du modèle entraîné
# --------------------------------------------------

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run train.py first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    saved_classes = checkpoint.get("class_names")

    if saved_classes != list(CLASS_NAMES):
        raise ValueError(
            f"Checkpoint classes: {saved_classes}. "
            f"Expected: {list(CLASS_NAMES)}"
        )

    model = build_model(
        num_classes=len(CLASS_NAMES),
        freeze_backbone=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


# --------------------------------------------------
# 4. Prédictions sur le test dataset
# --------------------------------------------------

def predict(model, test_loader):

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    return all_labels, all_predictions


# --------------------------------------------------
# 5. Calcul des métriques
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    return accuracy, report, matrix


# --------------------------------------------------
# 6. Sauvegarde des résultats
# --------------------------------------------------

def save_results(accuracy, report, matrix):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Rapport détaillé
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("MODEL EVALUATION\n")
        file.write("================\n\n")

        file.write(f"Device: {DEVICE}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")

        file.write("CLASSIFICATION REPORT\n")
        file.write("---------------------\n")

        file.write(report)

    # Matrice de confusion CSV
    with open(
        CONFUSION_MATRIX_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Actual / Predicted"] + CLASS_NAMES
        )

        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow(
                [class_name] + row.tolist()
            )


# --------------------------------------------------
# 7. Programme principal
# --------------------------------------------------

def main():

    print("MODEL EVALUATION")
    print("================")

    print("\nDevice:", DEVICE)

    print("\nLoading test dataset...")

    test_loader = load_test_dataset()

    print("Test images:", len(test_loader.dataset))
    print("Classes:", CLASS_NAMES)

    print("\nLoading trained model...")

    model, checkpoint = load_trained_model()

    print("Model loaded successfully.")
    print("Best training epoch:", checkpoint["epoch"])

    print("\nRunning predictions...")

    y_true, y_pred = predict(
        model,
        test_loader
    )

    accuracy, report, matrix = calculate_metrics(
        y_true,
        y_pred
    )

    print("\nFINAL TEST ACCURACY")
    print("-------------------")
    print(f"{accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print("---------------------")
    print(report)

    print("\nCONFUSION MATRIX")
    print("----------------")
    print("Rows = Actual classes")
    print("Columns = Predicted classes")
    print(matrix)

    save_results(
        accuracy,
        report,
        matrix
    )

    print("\nEvaluation completed.")

    print("Report:", REPORT_PATH)
    print("Confusion matrix:", CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
