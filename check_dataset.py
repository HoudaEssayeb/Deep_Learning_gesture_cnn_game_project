
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

# Seuil indicatif de netteté.
# Une valeur faible signale une image à examiner,
# mais ne signifie pas automatiquement qu'elle est inutilisable.
BLUR_THRESHOLD = 50.0

# Taille standard utilisée uniquement pour calculer
# la netteté de manière plus cohérente entre résolutions.
SHARPNESS_SIZE = (300, 300)

# Nombre maximal d'images affichées par planche.
MAX_CONTACT_IMAGES = 50

# Objectifs de collecte.
MIN_IMAGES_PER_CLASS = 300
TARGET_IMAGES_PER_CLASS = 500

# Fichier de rapport.
REPORT_PATH = REPORT_DIR / "quality_report.csv"


# ==========================================
# Utilities
# ==========================================

def calculate_hash(image_path):
    """
    Calculate SHA-256 to identify exact duplicate files.
    """

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_sharpness(image):
    """
    Estimate image sharpness using Laplacian variance.

    The image is resized to a standard size before
    calculating sharpness to reduce resolution-related
    differences.

    This score is indicative, not an absolute quality measure.
    """

    if image is None:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    normalized = cv2.resize(
        gray,
        SHARPNESS_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return float(
        cv2.Laplacian(
            normalized,
            cv2.CV_64F
        ).var()
    )


def create_contact_sheet(class_name, images):
    """
    Create a visual sheet containing sample images.
    """

    if not images:
        return

    thumbnail_width = 140
    thumbnail_height = 140
    label_height = 25
    columns = 5

    images = images[:MAX_CONTACT_IMAGES]

    rows = (
        len(images) + columns - 1
    ) // columns

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

    CONTACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = CONTACT_DIR / f"{class_name}.jpg"

    cv2.imwrite(
        str(output_path),
        sheet
    )

    print(f"[CONTACT SHEET] {output_path}")


def add_issue(
    issues,
    class_name,
    image_path,
    issue,
    sharpness="",
    duplicate_of_class="",
    duplicate_of_filename=""
):
    """
    Add an issue to the quality report.
    """

    issues.append({
        "class": class_name,
        "filename": image_path.name,
        "issue": issue,
        "sharpness": sharpness,
        "duplicate_of_class": duplicate_of_class,
        "duplicate_of_filename": duplicate_of_filename
    })


# ==========================================
# Dataset inspection
# ==========================================

def inspect_dataset():

    if not DATASET_DIR.exists():

        print(
            f"[ERROR] Dataset directory not found: {DATASET_DIR}"
        )

        return

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_files = 0
    total_readable = 0
    total_duplicates = 0
    total_blurry = 0
    total_invalid = 0
    total_cross_class_duplicates = 0

    class_counts = {}

    issues = []

    # SHA-256 -> first image location.
    hashes = {}

    print("\nDATASET QUALITY REPORT")
    print("======================")

    # ======================================
    # Inspect each class
    # ======================================

    for class_name in CLASS_NAMES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():

            print(
                f"\n[ERROR] Missing class directory: {class_name}"
            )

            class_counts[class_name] = 0

            continue

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_files += len(image_paths)

        valid_images = []

        class_counts[class_name] = 0

        class_duplicates = 0
        class_blurry = 0
        class_invalid = 0

        print(f"\nClass: {class_name}")
        print("-" * 40)

        for image_path in image_paths:

            # Read original channels without automatically
            # converting grayscale or alpha images to BGR.
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED
            )

            # ----------------------------------
            # 1. Readability
            # ----------------------------------

            if image is None:

                add_issue(
                    issues,
                    class_name,
                    image_path,
                    "Unreadable image"
                )

                class_invalid += 1
                total_invalid += 1

                print(f"[INVALID] {image_path.name}")

                continue

            # ----------------------------------
            # 2. Color format
            # ----------------------------------

            if len(image.shape) != 3 or image.shape[2] != 3:

                add_issue(
                    issues,
                    class_name,
                    image_path,
                    "Not a 3-channel BGR color image"
                )

                class_invalid += 1
                total_invalid += 1

                print(
                    f"[INVALID FORMAT] {image_path.name}"
                )

                continue

            # ----------------------------------
            # 3. Exact duplicate detection
            # ----------------------------------

            image_hash = calculate_hash(image_path)

            if image_hash in hashes:

                previous_class, previous_filename = hashes[image_hash]

                class_duplicates += 1
                total_duplicates += 1

                if previous_class != class_name:

                    duplicate_type = "Exact duplicate across classes"

                    total_cross_class_duplicates += 1

                    print(
                        f"[LABEL CONFLICT] {class_name}/{image_path.name} "
                        f"= {previous_class}/{previous_filename}"
                    )

                else:

                    duplicate_type = "Exact duplicate within same class"

                    print(
                        f"[DUPLICATE] {class_name}/{image_path.name} "
                        f"= {previous_filename}"
                    )

                add_issue(
                    issues,
                    class_name,
                    image_path,
                    duplicate_type,
                    duplicate_of_class=previous_class,
                    duplicate_of_filename=previous_filename
                )

            else:

                hashes[image_hash] = (
                    class_name,
                    image_path.name
                )

            # ----------------------------------
            # 4. Blur detection
            # ----------------------------------

            sharpness = calculate_sharpness(image)

            if sharpness < BLUR_THRESHOLD:

                class_blurry += 1
                total_blurry += 1

                add_issue(
                    issues,
                    class_name,
                    image_path,
                    "Potentially blurry image",
                    sharpness=round(sharpness, 2)
                )

                print(
                    f"[BLUR WARNING] {image_path.name} "
                    f"| Sharpness: {sharpness:.2f}"
                )

            # ----------------------------------
            # 5. Valid image
            # ----------------------------------

            valid_images.append(image_path)

            class_counts[class_name] += 1
            total_readable += 1

        # ----------------------------------
        # Class summary
        # ----------------------------------

        print(f"Total files: {len(image_paths)}")

        print(
            f"Readable color images: {class_counts[class_name]}"
        )

        print(f"Potentially blurry: {class_blurry}")
        print(f"Exact duplicates: {class_duplicates}")
        print(f"Invalid images: {class_invalid}")

        # Generate contact sheet.
        create_contact_sheet(
            class_name,
            valid_images
        )

    # ======================================
    # Class balance report
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

        print(
            f"{class_name:10} {count:5} images | {status}"
        )

    nonzero_counts = [
        count
        for count in class_counts.values()
        if count > 0
    ]

    if len(nonzero_counts) == len(CLASS_NAMES):

        minimum = min(nonzero_counts)
        maximum = max(nonzero_counts)

        difference = maximum - minimum

        print(
            f"\nDifference between largest and smallest class: "
            f"{difference}"
        )

        if difference <= max(5, int(maximum * 0.10)):

            print(
                "[BALANCE] Classes are reasonably balanced."
            )

        else:

            print(
                "[WARNING] Classes are imbalanced."
            )

    # ======================================
    # Save quality report
    # ======================================

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "class",
            "filename",
            "issue",
            "sharpness",
            "duplicate_of_class",
            "duplicate_of_filename"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(issues)

    # ======================================
    # Final summary
    # ======================================

    print("\nFINAL SUMMARY")
    print("=============")

    print(f"Total image files: {total_files}")
    print(f"Readable color images: {total_readable}")
    print(f"Invalid images: {total_invalid}")
    print(f"Potentially blurry images: {total_blurry}")
    print(f"Exact duplicates: {total_duplicates}")

    print(
        f"Exact duplicates across classes: "
        f"{total_cross_class_duplicates}"
    )

    print(f"Detected issue records: {len(issues)}")

    print(f"\nQuality report: {REPORT_PATH}")
    print(f"Contact sheets: {CONTACT_DIR}")

    print("\nIMPORTANT")
    print("=========")

    print(
        "- Different image dimensions are accepted."
    )

    print(
        "- Blur warnings require visual verification."
    )

    print(
        "- Exact duplicates should be reviewed."
    )

    print(
        "- Cross-class duplicates may indicate label conflicts."
    )

    print(
        "- No original images were deleted or modified."
    )

    print("\nInspection completed.")


# ==========================================
# Entry point
# ==========================================

if __name__ == "__main__":

    inspect_dataset()
