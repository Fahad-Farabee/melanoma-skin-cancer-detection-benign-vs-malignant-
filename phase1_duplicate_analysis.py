import os
import hashlib
import csv
from collections import defaultdict
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:/university/summer 26/Data Mining/research project/archive"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

OUTPUT_CSV = "phase1_duplicate_report.csv"


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# SHA-256 HASH
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_phash(file_path):

    with Image.open(file_path) as img:

        # Convert to grayscale
        img = img.convert("L")

        # Resize to 8x8
        img = img.resize((8, 8))

        pixels = list(img.getdata())

        average = sum(pixels) / len(pixels)

        hash_bits = ""

        for pixel in pixels:

            if pixel >= average:
                hash_bits += "1"
            else:
                hash_bits += "0"

        return int(hash_bits, 2)


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(hash1, hash2):

    return bin(hash1 ^ hash2).count("1")


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(split_path, split_name):

    images = []

    for class_name in ["Benign", "Malignant"]:

        class_path = os.path.join(split_path, class_name)

        if not os.path.exists(class_path):

            print(
                f"WARNING: {class_path} does not exist."
            )

            continue

        for filename in os.listdir(class_path):

            file_path = os.path.join(
                class_path,
                filename
            )

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            images.append({
                "path": file_path,
                "split": split_name,
                "class": class_name,
                "filename": filename
            })

    return images


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("PHASE 1 — DUPLICATE & LEAKAGE ANALYSIS")
print("=" * 70)


print("\nCollecting images...")

train_images = collect_images(
    TRAIN_PATH,
    "train"
)

test_images = collect_images(
    TEST_PATH,
    "test"
)

all_images = train_images + test_images

print(f"\nTrain images: {len(train_images)}")
print(f"Test images:  {len(test_images)}")
print(f"Total images: {len(all_images)}")


# ============================================================
# STEP 1 — EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — EXACT DUPLICATE CHECK")
print("=" * 70)

hash_groups = defaultdict(list)

for index, image in enumerate(all_images):

    try:

        file_hash = calculate_sha256(
            image["path"]
        )

        hash_groups[file_hash].append(image)

    except Exception as e:

        print(
            f"Could not process: "
            f"{image['path']}"
        )

    if (index + 1) % 1000 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(all_images)} images..."
        )


duplicate_groups = []

for file_hash, images in hash_groups.items():

    if len(images) > 1:

        duplicate_groups.append(
            {
                "hash": file_hash,
                "images": images
            }
        )


duplicate_image_count = sum(
    len(group["images"])
    for group in duplicate_groups
)


print(
    f"\nExact duplicate groups: "
    f"{len(duplicate_groups)}"
)

print(
    f"Images involved in exact duplicates: "
    f"{duplicate_image_count}"
)


# ============================================================
# TRAIN-TEST EXACT DUPLICATES
# ============================================================

train_test_duplicates = []

for group in duplicate_groups:

    splits = {
        image["split"]
        for image in group["images"]
    }

    if "train" in splits and "test" in splits:

        train_test_duplicates.append(group)


print(
    f"\nTrain-Test exact duplicate groups: "
    f"{len(train_test_duplicates)}"
)


# ============================================================
# SAVE EXACT DUPLICATE REPORT
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "Hash",
        "Split",
        "Class",
        "Filename",
        "Path"
    ])

    for group in duplicate_groups:

        for image in group["images"]:

            writer.writerow([
                group["hash"],
                image["split"],
                image["class"],
                image["filename"],
                image["path"]
            ])


print(
    f"\nDuplicate report saved to:"
    f"\n{OUTPUT_CSV}"
)


# ============================================================
# STEP 2 — PERCEPTUAL HASH
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — NEAR-DUPLICATE CHECK")
print("=" * 70)

print(
    "\nCalculating perceptual hashes..."
)


phashes = []

for index, image in enumerate(all_images):

    try:

        p_hash = calculate_phash(
            image["path"]
        )

        phashes.append({
            "hash": p_hash,
            "split": image["split"],
            "class": image["class"],
            "filename": image["filename"],
            "path": image["path"]
        })

    except Exception:

        pass

    if (index + 1) % 1000 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(all_images)} images..."
        )


# ============================================================
# COMPARE TRAIN VS TEST
# ============================================================

print(
    "\nSearching for potentially similar "
    "train/test images..."
)

near_duplicates = []

train_phashes = [
    item
    for item in phashes
    if item["split"] == "train"
]

test_phashes = [
    item
    for item in phashes
    if item["split"] == "test"
]


# A distance of 5 or less is used as a candidate threshold.
# These are candidates for manual inspection, NOT confirmed
# duplicates.

THRESHOLD = 5

for train_item in train_phashes:

    for test_item in test_phashes:

        distance = hamming_distance(
            train_item["hash"],
            test_item["hash"]
        )

        if distance <= THRESHOLD:

            near_duplicates.append({

                "train_file":
                    train_item["filename"],

                "train_class":
                    train_item["class"],

                "train_path":
                    train_item["path"],

                "test_file":
                    test_item["filename"],

                "test_class":
                    test_item["class"],

                "test_path":
                    test_item["path"],

                "hamming_distance":
                    distance
            })


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DUPLICATE ANALYSIS RESULTS")
print("=" * 70)

print(
    f"\nTotal images analyzed: "
    f"{len(all_images)}"
)

print(
    f"Exact duplicate groups: "
    f"{len(duplicate_groups)}"
)

print(
    f"Train-Test exact duplicate groups: "
    f"{len(train_test_duplicates)}"
)

print(
    f"Potential near-duplicate pairs: "
    f"{len(near_duplicates)}"
)


# ============================================================
# SAVE NEAR-DUPLICATE REPORT
# ============================================================

near_duplicate_csv = (
    "phase1_near_duplicate_candidates.csv"
)

with open(
    near_duplicate_csv,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "Train Filename",
        "Train Class",
        "Train Path",
        "Test Filename",
        "Test Class",
        "Test Path",
        "Hamming Distance"
    ])

    for item in near_duplicates:

        writer.writerow([
            item["train_file"],
            item["train_class"],
            item["train_path"],
            item["test_file"],
            item["test_class"],
            item["test_path"],
            item["hamming_distance"]
        ])


print(
    f"\nNear-duplicate candidate report saved to:"
    f"\n{near_duplicate_csv}"
)


print("\nAnalysis completed.")