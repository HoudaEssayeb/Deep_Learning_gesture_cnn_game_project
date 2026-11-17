
from pathlib import Path
import argparse
import uuid

# ==========================================
# Configuration
# ==========================================

RAW_DIR = Path("dataset/raw")

CLASS_NAMES = [
    "idle",
    "jump",
    "left",
    "right",
    "roll"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def rename_class_images(class_name, apply=False):
    class_dir = RAW_DIR / class_name

    if not class_dir.exists():
        print(f"[WARNING] Folder not found: {class_dir}")
        return

    images = sorted(
        file for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"[INFO] No images in {class_name}")
        return

    # Generate new names
    operations = []

    for index, source in enumerate(images, start=1):
        new_name = f"{class_name}_{index:04d}{source.suffix.lower()}"
        destination = class_dir / new_name

        if source.name != new_name:
            operations.append((source, destination))

    print(f"\nClass: {class_name}")
    print(f"Images: {len(images)}")
    print(f"Files to rename: {len(operations)}")

    if not operations:
        print("[INFO] Already correctly named.")
        return

    for source, destination in operations:
        print(f"{source.name} -> {destination.name}")

    if not apply:
        return

    # First phase: temporary names to avoid collisions
    temporary_operations = []

    for source, destination in operations:
        temporary = source.with_name(
            f".tmp_{uuid.uuid4().hex}{source.suffix}"
        )
        source.rename(temporary)
        temporary_operations.append((temporary, destination))

    # Second phase: final names
    for temporary, destination in temporary_operations:
        temporary.rename(destination)

    print("[SUCCESS] Renaming completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename the images."
    )
    args = parser.parse_args()

    print("DATASET IMAGE RENAMING")
    print("======================")

    if not args.apply:
        print("DRY RUN: No files will be modified.")
        print("Use --apply to execute the renaming.")

    for class_name in CLASS_NAMES:
        rename_class_images(class_name, apply=args.apply)

    print("\nRenaming inspection completed.")


if __name__ == "__main__":
    main()
