
from pathlib import Path

import cv2

from src.camera.camera import Camera


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = Path("dataset/raw")

CLASS_NAMES = [
    "idle",
    "jump",
    "left",
    "right",
    "roll"
]

KEY_TO_CLASS = {
    ord("1"): "idle",
    ord("2"): "jump",
    ord("3"): "left",
    ord("4"): "right",
    ord("5"): "roll"
}

ROI_SIZE = 300

WINDOW_NAME = "Gesture Dataset Collection"


# ==========================================
# Dataset management
# ==========================================

def create_dataset_directories():
    """Create a directory for each gesture class."""

    for class_name in CLASS_NAMES:
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)


def get_image_count(class_name):
    """Count images already saved for a class."""

    class_dir = DATASET_DIR / class_name

    extensions = ("*.jpg", "*.jpeg", "*.png")

    images = []

    for extension in extensions:
        images.extend(class_dir.glob(extension))

    return len(images)


def get_next_image_number(class_name):
    """Return the next available image number."""

    class_dir = DATASET_DIR / class_name

    existing_images = list(class_dir.glob(f"{class_name}_*.jpg"))

    numbers = []

    for image_path in existing_images:
        try:
            number = int(image_path.stem.split("_")[-1])
            numbers.append(number)
        except ValueError:
            continue

    return max(numbers, default=0) + 1


def save_image(class_name, image):
    """Save a captured image in its class directory."""

    image_number = get_next_image_number(class_name)

    filename = f"{class_name}_{image_number:04d}.jpg"

    image_path = DATASET_DIR / class_name / filename

    success = cv2.imwrite(
        str(image_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if success:
        print(f"[SAVED] {image_path}")
        return True

    print("[ERROR] Unable to save image.")
    return False


# ==========================================
# User interface
# ==========================================

def draw_information(frame, selected_class):
    """Display instructions and dataset counters."""

    display = frame.copy()

    if selected_class:
        status = f"Selected: {selected_class.upper()}"
    else:
        status = "Select a gesture (1-5)"

    cv2.putText(
        display,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    instructions = [
        "1: IDLE   2: JUMP   3: LEFT",
        "4: RIGHT  5: ROLL",
        "SPACE: CAPTURE",
        "Q: QUIT"
    ]

    y = 65

    for instruction in instructions:
        cv2.putText(
            display,
            instruction,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        y += 25

    # Display class counters
    y = 185

    for class_name in CLASS_NAMES:
        count = get_image_count(class_name)

        text = f"{class_name}: {count}"

        color = (
            (0, 255, 0)
            if class_name == selected_class
            else (255, 255, 255)
        )

        cv2.putText(
            display,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

        y += 25

    return display


# ==========================================
# Main
# ==========================================

def main():

    create_dataset_directories()

    camera = None

    selected_class = None

    try:
        camera = Camera(
            index=0,
            width=640,
            height=480
        )

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        print("\nGESTURE DATASET COLLECTION")
        print("--------------------------")
        print("1: idle")
        print("2: jump")
        print("3: left")
        print("4: right")
        print("5: roll")
        print("SPACE: capture image")
        print("Q: quit\n")

        while True:

            success, frame = camera.read()

            if not success:
                print("[ERROR] Unable to capture frame.")
                break

            height, width = frame.shape[:2]

            # Define a centered square ROI
            roi_size = min(
                ROI_SIZE,
                width,
                height
            )

            x1 = (width - roi_size) // 2
            y1 = (height - roi_size) // 2

            x2 = x1 + roi_size
            y2 = y1 + roi_size

            # Extract hand region
            roi = frame[y1:y2, x1:x2]

            # Draw capture zone
            display = frame.copy()

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            display = draw_information(
                display,
                selected_class
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            # Select gesture
            if key in KEY_TO_CLASS:
                selected_class = KEY_TO_CLASS[key]

                print(
                    f"[SELECTED] {selected_class.upper()}"
                )

            # Capture image
            elif key == ord(" "):

                if selected_class is None:
                    print("[WARNING] Select a gesture first.")
                    continue

                save_image(
                    selected_class,
                    roi.copy()
                )

            # Exit
            elif key == ord("q"):
                break

            # Detect window closure
            if cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    except Exception as error:
        print(f"[ERROR] {error}")

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("\nDataset collection stopped.")

        print("\nFinal dataset:")
        for class_name in CLASS_NAMES:
            count = get_image_count(class_name)
            print(f"{class_name}: {count} images")


if __name__ == "__main__":
    main()
