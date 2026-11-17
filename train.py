
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from src.model.cnn_model import build_model, CLASS_NAMES


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "mobilenetv2_best.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

NUM_WORKERS = 0  # Adapted for Windows

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. Vérification du dataset
# --------------------------------------------------

def load_datasets():
    weights = models.MobileNet_V2_Weights.DEFAULT

    # Augmentation uniquement pour les images d'entraînement.
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),
        weights.transforms()
    ])

    # Pas d'augmentation pour validation et test.
    evaluation_transform = weights.transforms()

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR / "val",
        transform=evaluation_transform
    )

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=evaluation_transform
    )

    # Vérifier que les classes sont identiques et dans le bon ordre.
    expected_classes = list(CLASS_NAMES)

    for name, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset)
    ]:
        if dataset.classes != expected_classes:
            raise ValueError(
                f"Classes incorrectes dans {name}: "
                f"{dataset.classes}. "
                f"Attendu: {expected_classes}"
            )

        if len(dataset) == 0:
            raise ValueError(f"Le dataset {name} est vide.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# 3. Entraînement d'une époque
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    # Garder aussi les BatchNorm du backbone en mode évaluation.
    model.features.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Remise à zéro des gradients.
        optimizer.zero_grad()

        # Prédiction du modèle.
        outputs = model(images)

        # Calcul de l'erreur.
        loss = criterion(outputs, labels)

        # Rétropropagation.
        loss.backward()

        # Mise à jour des poids.
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# --------------------------------------------------
# 4. Évaluation sur validation ou test
# --------------------------------------------------

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# --------------------------------------------------
# 5. Programme principal
# --------------------------------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Loading dataset...")

    train_loader, val_loader, test_loader = load_datasets()

    print("Classes:", CLASS_NAMES)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    print("\nBuilding MobileNetV2...")

    model = build_model(
        num_classes=len(CLASS_NAMES),
        freeze_backbone=True
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Seuls les paramètres entraînables sont optimisés.
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_val_accuracy = -1.0
    history = []

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy * 100:.2f}%"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Sauvegarder uniquement le meilleur modèle.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "val_accuracy": val_accuracy
            }, MODEL_PATH)

            print("  -> Best model saved.")

    # Sauvegarder l'historique des entraînements.
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ]
        )
        writer.writeheader()
        writer.writerows(history)

    # --------------------------------------------------
    # 6. Évaluation finale sur le test set
    # --------------------------------------------------

    print("\nLoading best model for final evaluation...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\nFinal test results")
    print("------------------")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_accuracy'] * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    print("\nModel saved at:", MODEL_PATH)
    print("Training history saved at:", HISTORY_PATH)


if __name__ == "__main__":
    main()
