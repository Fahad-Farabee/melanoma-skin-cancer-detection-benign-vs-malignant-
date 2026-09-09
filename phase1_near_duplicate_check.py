import os
import csv
import numpy as np
from PIL import Image
from scipy.fftpack import dct


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:/university/summer 26/Data Mining/research project/archive"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

OUTPUT_CSV = os.path.join(
    DATASET_PATH,
    "phase1_near_duplicate_high_confidence.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# pHASH THRESHOLD
# ============================================================

# Lower distance = more similar
#
# 0-2  = Very strong similarity
# 3-4  = Strong similarity
# 5-6  = Candidate
#
# We will use 6 as the maximum candidate threshold.

HAMMING_THRESHOLD = 6

# Maximum number of candidate pairs saved
MAX_RESULTS = 200


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(split_path, split_name):

    images = []

    for class_name in ["Benign", "Malignant"]:

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(class_path):

            raise FileNotFoundError(
                f"Folder not found:\n{class_path}"
            )

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
                "filename": filename,
                "split": split_name,
                "class": class_name
            })

    return images


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_phash(file_path):

    with Image.open(file_path) as img:

        # Convert to grayscale
        img = img.convert("L")

        # Resize to 32 x 32
        img = img.resize(
            (32, 32),
            Image.Resampling.LANCZOS
        )

        pixels = np.asarray(
            img,
            dtype=np.float32
        )

        # 2D DCT
        dct_rows = dct(
            pixels,
            axis=0,
            norm="ortho"
        )

        dct_2d = dct(
            dct_rows,
            axis=1,
            norm="ortho"
        )

        # Keep low-frequency information
        low_frequency = dct_2d[:8, :8]

        values = low_frequency.flatten()

        # Ignore DC coefficient when calculating median
        median = np.median(values[1:])

        # Generate binary hash
        hash_bits = (
            low_frequency > median
        )

        return hash_bits.flatten()


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(hash_a, hash_b):

    return int(
        np.count_nonzero(
            hash_a != hash_b
        )
    )


# ============================================================
# CALCULATE HASHES
# ============================================================

def calculate_hashes(images, name):

    hashes = []

    print(
        f"\nCalculating pHash for {name}..."
    )

    for index, image in enumerate(
        images,
        start=1
    ):

        try:

            image_hash = calculate_phash(
                image["path"]
            )

            hashes.append({
                "hash": image_hash,
                "path": image["path"],
                "filename": image["filename"],
                "split": image["split"],
                "class": image["class"]
            })

        except Exception as e:

            print(
                f"WARNING: Could not process:"
                f"\n{image['path']}"
                f"\nReason: {e}"
            )

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index}/{len(images)}..."
            )

    return hashes


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print(
    "PHASE 1 — HIGH-CONFIDENCE "
    "NEAR-DUPLICATE CHECK"
)
print("=" * 70)


# ============================================================
# COLLECT DATA
# ============================================================

train_images = collect_images(
    TRAIN_PATH,
    "train"
)

test_images = collect_images(
    TEST_PATH,
    "test"
)

print(
    f"\nTrain images: {len(train_images)}"
)

print(
    f"Test images:  {len(test_images)}"
)

print(
    f"Total images: "
    f"{len(train_images) + len(test_images)}"
)


# ============================================================
# HASH IMAGES
# ============================================================

train_hashes = calculate_hashes(
    train_images,
    "TRAIN"
)

test_hashes = calculate_hashes(
    test_images,
    "TEST"
)


# ============================================================
# TRAIN-TEST COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TRAIN-TEST NEAR-DUPLICATE SEARCH")
print("=" * 70)

print(
    "\nComparing test images against "
    "training images..."
)

candidates = []


for test_index, test_item in enumerate(
    test_hashes,
    start=1
):

    best_matches = []

    for train_item in train_hashes:

        distance = hamming_distance(
            test_item["hash"],
            train_item["hash"]
        )

        if distance <= HAMMING_THRESHOLD:

            best_matches.append({

                "distance": distance,

                "train": train_item,

                "test": test_item

            })

    # Sort strongest matches first
    best_matches.sort(
        key=lambda x: x["distance"]
    )

    # Keep only strongest 10 matches
    candidates.extend(
        best_matches[:10]
    )

    if test_index % 100 == 0:

        print(
            f"Compared "
            f"{test_index}/{len(test_hashes)} "
            f"test images..."
        )


# ============================================================
# SORT GLOBAL RESULTS
# ============================================================

candidates.sort(
    key=lambda x: x["distance"]
)


# Keep only strongest candidates
candidates = candidates[:MAX_RESULTS]


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Hamming_Distance",
        "Similarity_Strength",
        "Train_Class",
        "Train_Filename",
        "Train_Path",
        "Test_Class",
        "Test_Filename",
        "Test_Path"
    ])

    for item in candidates:

        distance = item["distance"]

        if distance <= 2:

            strength = "Very Strong"

        elif distance <= 4:

            strength = "Strong"

        else:

            strength = "Candidate"

        writer.writerow([
            distance,
            strength,
            item["train"]["class"],
            item["train"]["filename"],
            item["train"]["path"],
            item["test"]["class"],
            item["test"]["filename"],
            item["test"]["path"]
        ])


# ============================================================
# SUMMARY
# ============================================================

very_strong = sum(
    1
    for item in candidates
    if item["distance"] <= 2
)

strong = sum(
    1
    for item in candidates
    if item["distance"] <= 4
)

same_class = sum(
    1
    for item in candidates
    if (
        item["train"]["class"]
        ==
        item["test"]["class"]
    )
)

different_class = sum(
    1
    for item in candidates
    if (
        item["train"]["class"]
        !=
        item["test"]["class"]
    )
)


print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"\nCandidate pairs saved: "
    f"{len(candidates)}"
)

print(
    f"Very strong candidates (0-2): "
    f"{very_strong}"
)

print(
    f"Strong candidates (0-4): "
    f"{strong}"
)

print(
    f"Same-class candidates: "
    f"{same_class}"
)

print(
    f"Different-class candidates: "
    f"{different_class}"
)

print("\nIMPORTANT:")
print(
    "These are similarity candidates, "
    "NOT confirmed duplicates."
)

print(
    "\nThey must be visually inspected "
    "before any image is removed."
)

print(
    f"\nReport saved to:\n{OUTPUT_CSV}"
)

print("\nAnalysis completed.")