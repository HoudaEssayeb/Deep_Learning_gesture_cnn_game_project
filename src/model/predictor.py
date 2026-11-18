
from pathlib import Path
import time

import cv2
import torch

from src.camera.camera import Camera
from src.model.cnn_model import build_model, CLASS_NAMES
from src.model.preprocessing import ImagePreprocessor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
# Limit CPU threads to avoid excessive overhead
torch.set_num_threads(2)

MODEL_PATH = Path("models/mobilenetv2_best.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROI_SIZE = 300


# --------------------------------------------------
# 2. Gesture Predictor
# --------------------------------------------------

class GesturePredictor:
    """Loads MobileNetV2 and predicts hand gestures."""

    def __init__(self, model_path=MODEL_PATH):

        self.device = DEVICE
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        print("Loading gesture recognition model...")

        # Load checkpoint.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True
        )

        # Verify class order.
        saved_classes = checkpoint.get("class_names")

        if saved_classes != list(CLASS_NAMES):
            raise ValueError(
                f"Checkpoint classes: {saved_classes}. "
                f"Expected: {list(CLASS_NAMES)}"
            )

        # Build model architecture.
        self.model = build_model(
            num_classes=len(CLASS_NAMES),
            freeze_backbone=True
        )

        # Load trained weights.
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        # Reuse the training preprocessing.
        self.preprocessor = ImagePreprocessor()

        self.class_names = list(CLASS_NAMES)

        print("Model loaded successfully.")
        print("Device:", self.device)
        print("Classes:", self.class_names)

    # --------------------------------------------------
    # 3. Extract hand region (ROI)
    # --------------------------------------------------

    @staticmethod
    def extract_roi(frame):

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        height, width = frame.shape[:2]

        x1 = (width - ROI_SIZE) // 2
        y1 = (height - ROI_SIZE) // 2

        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        roi = frame[y1:y2, x1:x2]

        if roi.shape[0] != ROI_SIZE or roi.shape[1] != ROI_SIZE:
            raise ValueError(
                "Unable to extract the expected ROI."
            )

        return roi, (x1, y1, x2, y2)

    # --------------------------------------------------
    # 4. Predict gesture
    # --------------------------------------------------

    def predict(self, roi):

        start_time = time.perf_counter()

        # Apply exactly the same preprocessing.
        input_tensor = self.preprocessor.process(roi)

        input_tensor = input_tensor.to(self.device)

        # Model inference.
        with torch.inference_mode():

            outputs = self.model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        gesture = self.class_names[predicted_index.item()]
        confidence_value = confidence.item()

        # Get the two most probable classes
        top_values, top_indices = torch.topk(
            probabilities,
            k=2,
            dim=1
        )

        top_predictions = [
            {
                "gesture": self.class_names[index],
                "confidence": value
            }
            for value, index in zip(
                top_values[0].tolist(),
                top_indices[0].tolist()
            )
        ]

        inference_time = (
            time.perf_counter() - start_time
        ) * 1000


        return {
            "gesture": gesture,
            "confidence": confidence_value,
            "top_predictions": top_predictions,
            "inference_time_ms": inference_time
        }
    
# --------------------------------------------------
# 5. Webcam demonstration
# --------------------------------------------------


def main():

    predictor = GesturePredictor()

    camera = Camera(
        index=0,
        width=640,
        height=480
    )

    print("\nReal-time gesture recognition started.")
    print("Press Q to quit.")

    # Predict once every 5 frames.
    inference_interval = 5
    frame_count = 0

    # Keep the last prediction between inference frames.
    last_result = {
        "gesture": "Detecting...",
        "confidence": 0.0,
        "inference_time_ms": 0.0
    }

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

            # Perform inference only every 5 frames.
            if frame_count % inference_interval == 0:

                roi, (x1, y1, x2, y2) = (
                    predictor.extract_roi(frame)
                )

                last_result = predictor.predict(roi)

            top_predictions = last_result["top_predictions"]

            top_text = " | ".join(
                f"{item['gesture']}: {item['confidence'] * 100:.1f}%"
                for item in top_predictions
            )

            print(
                f"Prediction: {last_result['gesture']} "
                f"| Top 2: {top_text}"
            )

            frame_count += 1

            gesture = last_result["gesture"]
            confidence = last_result["confidence"]
            inference_time = last_result["inference_time_ms"]

            # Display ROI.
            roi, (x1, y1, x2, y2) = (
                predictor.extract_roi(frame)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display prediction.
            cv2.putText(
                frame,
                f"{gesture} | {confidence * 100:.1f}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Display inference time.
            cv2.putText(
                frame,
                f"Inference: {inference_time:.1f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Real-Time Gesture Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nRecognition interrupted by user.")

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("\nCamera released.")
        print("Recognition stopped.")


if __name__ == "__main__":
    main()
