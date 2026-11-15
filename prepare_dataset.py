
from pathlib import Path
import random
import shutil
import hashlib

# ==========================================
# Configuration
# ==========================================

RAW_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/processed")

CLASS_NAMES = [
    "idle",
    "jump",
    "left",
    "right",
    "roll"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


# ==========================================
# Utilities
# ==========================================

def calculate_hash(image_path):
    """Calculate the SHA256 hash of an image."""

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def split_images(images):
    """Split images into train, validation and test sets."""

    images = images.copy()

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    return {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }


def create_output_directories():
    """Create output directories for all classes."""

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            (OUTPUT_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


# ==========================================
# Dataset preparation
# ==========================================

def prepare_dataset():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DIR}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Rename or remove it before preparing a new split."
        )

    random.seed(RANDOM_SEED)

    # Check classes
    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

    # Detect exact duplicates before splitting
    hashes = {}

    for class_name in CLASS_NAMES:

        class_dir = RAW_DIR / class_name

        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(
                f"No images found for class: {class_name}"
            )

        for image_path in images:

            image_hash = calculate_hash(image_path)

            if image_hash in hashes:
                previous = hashes[image_hash]

                raise ValueError(
                    "Exact duplicate detected. "
                    f"{image_path} is identical to {previous}. "
                    "Resolve duplicates before splitting."
                )

            hashes[image_hash] = image_path

    create_output_directories()

    print("\nDATASET SPLITTING")
    print("=================")

    total_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    # Split each class independently
    for class_name in CLASS_NAMES:

        class_dir = RAW_DIR / class_name

        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        splits = split_images(images)

        print(f"\nClass: {class_name}")
        print("-" * 35)

        for split_name, split_images_list in splits.items():

            destination_dir = OUTPUT_DIR / split_name / class_name

            for image_path in split_images_list:

                destination = destination_dir / image_path.name

                shutil.copy2(
                    image_path,
                    destination
                )

            count = len(split_images_list)

            total_counts[split_name] += count

            print(f"{split_name:10}: {count} images")

    # Summary
    total = sum(total_counts.values())

    print("\nFINAL SUMMARY")
    print("=============")

    for split_name, count in total_counts.items():

        percentage = (count / total) * 100

        print(
            f"{split_name:10}: "
            f"{count:5} images ({percentage:.1f}%)"
        )

    print(f"\nTotal images: {total}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    prepare_dataset()
