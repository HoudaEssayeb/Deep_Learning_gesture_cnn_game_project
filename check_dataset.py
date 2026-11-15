
from pathlib import Path
import hashlib
import csv

import cv2
import numpy as np


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = Path("dataset/raw")
REPORT_DIR = Path("dataset/reports")
CONTACT_DIR = REPORT_DIR / "contact_sheets"

CLASS_NAMES = [
    "idle",
    "jump",
    "left",
    "right",
    "roll"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}
EXPECTED_SIZE = (300, 300)

# Valeur indicative : à interpréter avec les images
BLUR_THRESHOLD = 50.0

# Nombre maximal d'images affichées par planche
MAX_CONTACT_IMAGES = 50

# Objectifs de collecte
MIN_IMAGES_PER_CLASS = 300
TARGET_IMAGES_PER_CLASS = 500


# ==========================================
# Utilities
# ==========================================

def calculate_hash(image_path):
    """Calculate SHA256 to identify exact duplicate files."""

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_sharpness(image):
    """Estimate image sharpness using Laplacian variance."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def create_contact_sheet(class_name, images):
    """Create a visual sheet containing sample images."""

    if not images:
        return

    thumbnail_width = 140
    thumbnail_height = 140
    label_height = 25
    columns = 5

    images = images[:MAX_CONTACT_IMAGES]

    rows = (len(images) + columns - 1) // columns

    sheet = np.full(
        (
            rows * (thumbnail_height + label_height),
            columns * thumbnail_width,
            3
        ),
        255,
        dtype=np.uint8
    )

    for index, image_path in enumerate(images):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        thumbnail = cv2.resize(
            image,
            (thumbnail_width, thumbnail_height)
        )

        row = index // columns
        column = index % columns

        x = column * thumbnail_width
        y = row * (thumbnail_height + label_height)

        sheet[
            y:y + thumbnail_height,
            x:x + thumbnail_width
        ] = thumbnail

        filename = image_path.name[:19]

        cv2.putText(
            sheet,
            filename,
            (x + 3, y + thumbnail_height + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1
        )

    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CONTACT_DIR / f"{class_name}.jpg"

    cv2.imwrite(str(output_path), sheet)

    print(f"[CONTACT SHEET] {output_path}")


# ==========================================
# Dataset inspection
# ==========================================

def inspect_dataset():

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_valid = 0

    class_counts = {}
    issues = []
    hashes = {}

    print("\nDATASET QUALITY REPORT")
    print("======================")

    for class_name in CLASS_NAMES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"\n[ERROR] Missing class directory: {class_name}")
            class_counts[class_name] = 0
            continue

        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_files += len(image_paths)

        valid_images = []
        class_counts[class_name] = 0

        print(f"\nClass: {class_name}")
        print("-" * 35)

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            if image is None:
                issues.append({
                    "class": class_name,
                    "filename": image_path.name,
                    "issue": "Unreadable image",
                    "sharpness": ""
                })
                print(f"[INVALID] {image_path.name}")
                continue

            height, width = image.shape[:2]

            if len(image.shape) != 3 or image.shape[2] != 3:
                issues.append({
                    "class": class_name,
                    "filename": image_path.name,
                    "issue": "Not a 3-channel color image",
                    "sharpness": ""
                })
                continue

            if (width, height) != EXPECTED_SIZE:
                issues.append({
                    "class": class_name,
                    "filename": image_path.name,
                    "issue": f"Unexpected dimensions: {width}x{height}",
                    "sharpness": ""
                })

            sharpness = calculate_sharpness(image)

            if sharpness < BLUR_THRESHOLD:
                issues.append({
                    "class": class_name,
                    "filename": image_path.name,
                    "issue": "Potentially blurry image",
                    "sharpness": round(sharpness, 2)
                })

            image_hash = calculate_hash(image_path)

            if image_hash in hashes:
                previous_class, previous_filename = hashes[image_hash]

                issues.append({
                    "class": class_name,
                    "filename": image_path.name,
                    "issue": (
                        f"Exact duplicate of "
                        f"{previous_class}/{previous_filename}"
                    ),
                    "sharpness": round(sharpness, 2)
                })
            else:
                hashes[image_hash] = (
                    class_name,
                    image_path.name
                )

            valid_images.append(image_path)
            class_counts[class_name] += 1
            total_valid += 1

        print(f"Total files: {len(image_paths)}")
        print(f"Valid readable images: {class_counts[class_name]}")

        create_contact_sheet(class_name, valid_images)

    # ======================================
    # Balance report
    # ======================================

    print("\nCLASS BALANCE")
    print("=============")

    for class_name, count in class_counts.items():

        if count < MIN_IMAGES_PER_CLASS:
            status = "INCOMPLETE"
        elif count < TARGET_IMAGES_PER_CLASS:
            status = "ACCEPTABLE - CAN BE ENRICHED"
        else:
            status = "TARGET REACHED"

        print(f"{class_name:10} {count:5} images | {status}")

    nonzero_counts = [
        count for count in class_counts.values()
        if count > 0
    ]

    if len(nonzero_counts) == len(CLASS_NAMES):

        minimum = min(nonzero_counts)
        maximum = max(nonzero_counts)

        difference = maximum - minimum

        print(f"\nDifference between largest and smallest class: {difference}")

        if difference <= max(5, int(maximum * 0.10)):
            print("[BALANCE] Classes are reasonably balanced.")
        else:
            print("[WARNING] Classes are imbalanced.")

    # ======================================
    # Issue report
    # ======================================

    report_path = REPORT_DIR / "quality_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "filename",
                "issue",
                "sharpness"
            ]
        )

        writer.writeheader()
        writer.writerows(issues)

    print("\nFINAL SUMMARY")
    print("=============")
    print(f"Total image files: {total_files}")
    print(f"Valid readable images: {total_valid}")
    print(f"Detected issues: {len(issues)}")
    print(f"Quality report: {report_path}")
    print(f"Contact sheets: {CONTACT_DIR}")

    if issues:
        print("\n[WARNING] Review the detected issues manually.")
        print("No original images were deleted or modified.")
    else:
        print("\nNo automatic quality issues detected.")

    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_dataset()
