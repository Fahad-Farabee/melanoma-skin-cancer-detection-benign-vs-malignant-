import os
from PIL import Image
from collections import Counter

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = r"D:/university/summer 26/Data Mining/research project/archive"

# ==========================================
# SUPPORTED IMAGE EXTENSIONS
# ==========================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# ANALYZE ONE SPLIT
# ==========================================

def analyze_split(split_path, split_name):

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} DATASET")
    print("=" * 60)

    total_images = 0
    class_counts = Counter()

    widths = []
    heights = []

    formats = Counter()
    modes = Counter()

    corrupted = []

    # --------------------------------------
    # Check classes
    # --------------------------------------

    for class_name in ["Benign", "Malignant"]:

        class_path = os.path.join(split_path, class_name)

        if not os.path.exists(class_path):
            print(f"WARNING: {class_name} folder not found.")
            continue

        for filename in os.listdir(class_path):

            file_path = os.path.join(class_path, filename)

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(filename)[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            total_images += 1
            class_counts[class_name] += 1

            try:

                with Image.open(file_path) as img:

                    widths.append(img.width)
                    heights.append(img.height)

                    formats[img.format] += 1
                    modes[img.mode] += 1

                    # Verify that the image can actually be read
                    img.verify()

            except Exception:

                corrupted.append(file_path)

    # ======================================
    # RESULTS
    # ======================================

    print(f"\nTotal images: {total_images}")

    print("\nClass distribution:")

    for class_name in ["Benign", "Malignant"]:

        count = class_counts[class_name]

        percentage = (
            (count / total_images) * 100
            if total_images > 0
            else 0
        )

        print(
            f"  {class_name}: "
            f"{count} images "
            f"({percentage:.2f}%)"
        )

    if widths:

        print("\nImage dimensions:")

        print(f"  Minimum width:  {min(widths)}")
        print(f"  Maximum width:  {max(widths)}")
        print(f"  Average width:  {sum(widths) / len(widths):.2f}")

        print(f"  Minimum height: {min(heights)}")
        print(f"  Maximum height: {max(heights)}")
        print(f"  Average height: {sum(heights) / len(heights):.2f}")

    print("\nImage formats:")

    for fmt, count in formats.items():
        print(f"  {fmt}: {count}")

    print("\nImage color modes:")

    for mode, count in modes.items():
        print(f"  {mode}: {count}")

    print(f"\nCorrupted images: {len(corrupted)}")

    if corrupted:

        print("\nCorrupted files:")

        for file_path in corrupted[:10]:
            print(f"  {file_path}")

        if len(corrupted) > 10:
            print(
                f"  ... and "
                f"{len(corrupted) - 10} more"
            )

    return total_images, class_counts


# ==========================================
# MAIN
# ==========================================

train_path = os.path.join(DATASET_PATH, "train")
test_path = os.path.join(DATASET_PATH, "test")

train_total, train_classes = analyze_split(
    train_path,
    "Train"
)

test_total, test_classes = analyze_split(
    test_path,
    "Test"
)

# ==========================================
# OVERALL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("OVERALL DATASET SUMMARY")
print("=" * 60)

total = train_total + test_total

print(f"\nTotal images: {total}")

print("\nTraining:")
print(f"  Benign:    {train_classes['Benign']}")
print(f"  Malignant: {train_classes['Malignant']}")

print("\nTesting:")
print(f"  Benign:    {test_classes['Benign']}")
print(f"  Malignant: {test_classes['Malignant']}")

print("\nAnalysis completed.")