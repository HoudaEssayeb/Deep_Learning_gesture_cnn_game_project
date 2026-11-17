from pathlib import Path

import customtkinter as ctk
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import MobileNet_V2_Weights

from sklearn.metrics import confusion_matrix, classification_report

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.model.cnn_model import build_model, CLASS_NAMES


MODEL_PATH = Path("models/mobilenetv2_best.pth")
HISTORY_PATH = Path("models/training_history.csv")
TEST_DIR = Path("dataset/processed/test")


class PerformanceView(ctk.CTkToplevel):
    """Displays training curves and test-set evaluation metrics."""

    def __init__(self, master):
        super().__init__(master)

        self.title("Model Performance Dashboard")
        self.geometry("1150x850")
        self.minsize(950, 700)

        self.transient(master)

        self._build_interface()

    def _build_interface(self):
        """Create the performance dashboard."""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        title = ctk.CTkLabel(
            self,
            text="MODEL PERFORMANCE DASHBOARD",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.grid(
            row=0,
            column=0,
            padx=20,
            pady=(20, 5),
            sticky="w"
        )

        subtitle = ctk.CTkLabel(
            self,
            text="Training history and evaluation on the test dataset",
            text_color="gray"
        )
        subtitle.grid(
            row=1,
            column=0,
            padx=20,
            pady=(0, 15),
            sticky="w"
        )

        # Main plotting area.
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(
            row=2,
            column=0,
            padx=20,
            pady=10,
            sticky="nsew"
        )
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

 
        # Metrics and actions.
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(
            row=3,
            column=0,
            padx=20,
            pady=(5, 15),
            sticky="ew"
        )
        bottom.grid_columnconfigure(0, weight=1)

        self.metrics_label = ctk.CTkLabel(
            bottom,
            text="Evaluation metrics will appear here.",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.metrics_label.grid(
            row=0,
            column=0,
            sticky="w",
            pady=5
        )

        self.class_metrics_label = ctk.CTkLabel(
            bottom,
            text="Per-class metrics will appear here.",
            text_color="gray",
            justify="left",
            anchor="w",
            wraplength=850
        )
        self.class_metrics_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=5
        )

        self.status_label = ctk.CTkLabel(
            bottom,
            text="Ready",
            text_color="gray"
        )
        self.status_label.grid(
            row=2,
            column=0,
            sticky="w",
            pady=5
        )

        self.refresh_button = ctk.CTkButton(
            bottom,
            text="REFRESH PERFORMANCE",
            command=self.refresh
        )
        self.refresh_button.grid(
            row=0,
            column=1,
            rowspan=3,
            padx=10
        )

        self.refresh()

    @staticmethod
    def _normalize_accuracy(values):
        """Convert accuracy values to percentages when necessary."""

        values = [float(value) for value in values]

        if not values:
            return []

        if max(values) <= 1.0:
            return [value * 100 for value in values]

        return values
        
    @staticmethod
    def _find_column(dataframe, candidates):
        """Find a CSV column using possible column-name aliases."""

        normalized = {
            str(column).strip().lower().replace(" ", "_"): column
            for column in dataframe.columns
        }

        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]

        return None

    @staticmethod
    def _load_training_history():
        """Load training history and identify its columns."""

        if not HISTORY_PATH.exists():
            raise FileNotFoundError(
                f"Training history not found: {HISTORY_PATH}"
            )

        history = pd.read_csv(HISTORY_PATH)

        if history.empty:
            raise ValueError("Training history CSV is empty.")

        epoch_col = PerformanceView._find_column(
            history,
            ["epoch"]
        )

        train_loss_col = PerformanceView._find_column(
            history,
            ["train_loss", "training_loss"]
        )

        val_loss_col = PerformanceView._find_column(
            history,
            ["val_loss", "validation_loss"]
        )

        train_acc_col = PerformanceView._find_column(
            history,
            ["train_accuracy", "train_acc", "training_accuracy"]
        )

        val_acc_col = PerformanceView._find_column(
            history,
            ["val_accuracy", "val_acc", "validation_accuracy"]
        )

        required = {
            "train_loss": train_loss_col,
            "val_loss": val_loss_col,
            "train_accuracy": train_acc_col,
            "val_accuracy": val_acc_col
        }

        missing = [
            name for name, column in required.items()
            if column is None
        ]

        if missing:
            raise ValueError(
                "Missing columns in training_history.csv: "
                + ", ".join(missing)
                + ". Found columns: "
                + ", ".join(map(str, history.columns))
            )

        epochs = (
            history[epoch_col].tolist()
            if epoch_col is not None
            else list(range(1, len(history) + 1))
        )

        return {
            "epochs": epochs,
            "train_loss": history[train_loss_col].tolist(),
            "val_loss": history[val_loss_col].tolist(),
            "train_accuracy": history[train_acc_col].tolist(),
            "val_accuracy": history[val_acc_col].tolist()
        }

    @staticmethod
    def _load_test_predictions():
        """Load the saved model and predict the test dataset."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}"
            )

        if not TEST_DIR.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {TEST_DIR}"
            )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )

        saved_classes = checkpoint.get("class_names")

        if saved_classes != list(CLASS_NAMES):
            raise ValueError(
                f"Checkpoint classes do not match: {saved_classes}"
            )

        weights = MobileNet_V2_Weights.DEFAULT
        transform = weights.transforms()

        test_dataset = datasets.ImageFolder(
            root=str(TEST_DIR),
            transform=transform
        )

        if test_dataset.classes != list(CLASS_NAMES):
            raise ValueError(
                f"Test dataset classes: {test_dataset.classes}. "
                f"Expected: {list(CLASS_NAMES)}"
            )

        if len(test_dataset) == 0:
            raise ValueError("The test dataset is empty.")

        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        model = build_model(
            num_classes=len(CLASS_NAMES),
            freeze_backbone=True
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        all_labels = []
        all_predictions = []

        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(device)

                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.cpu().tolist())

        return (
            all_labels,
            all_predictions,
            test_dataset.classes
        )

    @staticmethod
    def _normalize_accuracy(values):
        """Convert accuracy values to percentages when necessary."""

        values = [float(value) for value in values]

        if not values:
            return []

        if max(values) <= 1.0:
            return [value * 100 for value in values]

        return values

    def refresh(self):
        """Reload history, evaluate the model and redraw charts."""

        self.refresh_button.configure(state="disabled")
        self.status_label.configure(
            text="Loading training history and evaluating model...",
            text_color="#F59E0B"
        )

        try:
            history = self._load_training_history()

            labels, predictions, classes = (
                self._load_test_predictions()
            )

            matrix = confusion_matrix(
                labels,
                predictions,
                labels=list(range(len(classes)))
            )

            accuracy = sum(
                actual == predicted
                for actual, predicted in zip(labels, predictions)
            ) / len(labels) * 100

            report = classification_report(
                labels,
                predictions,
                labels=list(range(len(classes))),
                target_names=classes,
                output_dict=True,
                zero_division=0
            )

            self._draw_charts(
                history,
                matrix,
                classes
            )

            self.metrics_label.configure(
                text=(
                    f"Test accuracy: {accuracy:.1f}%"
                    f"     |     Test images: {len(labels)}"
                    f"     |     Classes: {len(classes)}"
                )
            )


            class_details = []

            for class_name in classes:
                scores = report[class_name]

                class_details.append(
                    f"{class_name}: "
                    f"Precision {scores['precision']:.0%} | "
                    f"Recall {scores['recall']:.0%} | "
                    f"F1 {scores['f1-score']:.0%}"
                )

            self.class_metrics_label.configure(
                text="     •     ".join(class_details)
            )

            self.status_label.configure(
                text="Evaluation completed successfully.",
                text_color="#22C55E"
            )

        except Exception as error:
            self.status_label.configure(
                text=f"Evaluation error: {error}",
                text_color="#EF4444"
            )

        finally:
            self.refresh_button.configure(state="normal")

    def _draw_charts(self, history, matrix, classes):
        """Draw training curves and the confusion matrix."""

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        figure = Figure(figsize=(12, 8), dpi=100)

        # Training and validation loss.
        ax1 = figure.add_subplot(221)

        ax1.plot(
            history["epochs"],
            history["train_loss"],
            marker="o",
            label="Training loss"
        )

        ax1.plot(
            history["epochs"],
            history["val_loss"],
            marker="o",
            label="Validation loss"
        )

        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Training and validation accuracy.
        ax2 = figure.add_subplot(222)

        train_accuracy = self._normalize_accuracy(
            history["train_accuracy"]
        )

        val_accuracy = self._normalize_accuracy(
            history["val_accuracy"]
        )

        ax2.plot(
            history["epochs"],
            train_accuracy,
            marker="o",
            label="Training accuracy"
        )

        ax2.plot(
            history["epochs"],
            val_accuracy,
            marker="o",
            label="Validation accuracy"
        )

        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Confusion matrix.
        ax3 = figure.add_subplot(212)

        image = ax3.imshow(
            matrix,
            interpolation="nearest",
            cmap="Blues"
        )

        figure.colorbar(image, ax=ax3, fraction=0.046, pad=0.04)

        ax3.set_title("Confusion Matrix - Test Dataset")
        ax3.set_xlabel("Predicted gesture")
        ax3.set_ylabel("Actual gesture")

        positions = range(len(classes))

        ax3.set_xticks(list(positions))
        ax3.set_yticks(list(positions))
        ax3.set_xticklabels(classes)
        ax3.set_yticklabels(classes)

        threshold = matrix.max() / 2 if matrix.size else 0

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax3.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row, column] > threshold
                        else "black"
                    )
                )

        figure.tight_layout(pad=3)

        canvas = FigureCanvasTkAgg(
            figure,
            master=self.plot_frame
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # Keep references to avoid premature garbage collection.
        self.figure = figure
        self.canvas = canvas
