import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PHASE 3.1
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

DATASET_ROOT = r"D:/university/summer 26/Data Mining/research project/archive"

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train"
)

TEST_DIR = os.path.join(
    DATASET_ROOT,
    "test"
)

OUTPUT_DIR = os.path.join(
    DATASET_ROOT,
    "phase3"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Label mapping
CLASS_TO_LABEL = {
    "Benign": 0,
    "Malignant": 1
}


# ============================================================
# COLLECT TRAINING IMAGES
# ============================================================

records = []

for class_name, label in CLASS_TO_LABEL.items():

    class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    if not os.path.exists(class_dir):
        raise FileNotFoundError(
            f"Directory not found: {class_dir}"
        )

    for filename in sorted(os.listdir(class_dir)):

        filepath = os.path.join(
            class_dir,
            filename
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        if (
            os.path.isfile(filepath)
            and extension in IMAGE_EXTENSIONS
        ):

            records.append({
                "filepath": filepath,
                "class": class_name,
                "label": label
            })


train_df = pd.DataFrame(records)


# ============================================================
# VERIFY ORIGINAL TRAINING DATA
# ============================================================

print("\n==========================================")
print("PHASE 3.1")
print("STRATIFIED TRAIN / VALIDATION SPLIT")
print("==========================================")

print(
    f"\nTotal training images found: "
    f"{len(train_df):,}"
)

print("\nOriginal class distribution:")

print(
    train_df["class"]
    .value_counts()
)


# ============================================================
# STRATIFIED SPLIT
# ============================================================

train_split, validation_split = train_test_split(
    train_df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_df["label"]
)


# Reset indexes
train_split = train_split.reset_index(drop=True)
validation_split = validation_split.reset_index(drop=True)


# ============================================================
# COLLECT ORIGINAL TEST SET
# ============================================================

test_records = []

for class_name, label in CLASS_TO_LABEL.items():

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.exists(class_dir):
        raise FileNotFoundError(
            f"Directory not found: {class_dir}"
        )

    for filename in sorted(os.listdir(class_dir)):

        filepath = os.path.join(
            class_dir,
            filename
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        if (
            os.path.isfile(filepath)
            and extension in IMAGE_EXTENSIONS
        ):

            test_records.append({
                "filepath": filepath,
                "class": class_name,
                "label": label
            })


test_df = pd.DataFrame(test_records)


# ============================================================
# SAVE MANIFESTS
# ============================================================

train_file = os.path.join(
    OUTPUT_DIR,
    "train_manifest.csv"
)

validation_file = os.path.join(
    OUTPUT_DIR,
    "validation_manifest.csv"
)

test_file = os.path.join(
    OUTPUT_DIR,
    "test_manifest.csv"
)


train_split.to_csv(
    train_file,
    index=False
)

validation_split.to_csv(
    validation_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# PRINT SPLIT RESULTS
# ============================================================

print("\n==========================================")
print("SPLIT RESULTS")
print("==========================================")

print(
    f"\nTraining set:    {len(train_split):,}"
)

print(
    f"Validation set: {len(validation_split):,}"
)

print(
    f"Test set:        {len(test_df):,}"
)

print(
    f"\nTotal: "
    f"{len(train_split) + len(validation_split) + len(test_df):,}"
)


# ============================================================
# CLASS DISTRIBUTIONS
# ============================================================

print("\n==========================================")
print("CLASS DISTRIBUTIONS")
print("==========================================")


def print_distribution(
    dataframe,
    name
):

    counts = (
        dataframe["class"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        dataframe["class"]
        .value_counts(
            normalize=True
        )
        .sort_index() * 100
    )

    print(f"\n{name}")

    for class_name in [
        "Benign",
        "Malignant"
    ]:

        count = counts.get(
            class_name,
            0
        )

        percentage = percentages.get(
            class_name,
            0
        )

        print(
            f"  {class_name:10s}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


print_distribution(
    train_split,
    "TRAINING"
)

print_distribution(
    validation_split,
    "VALIDATION"
)

print_distribution(
    test_df,
    "TEST"
)


# ============================================================
# VERIFY NO OVERLAP
# ============================================================

train_paths = set(
    train_split["filepath"]
)

validation_paths = set(
    validation_split["filepath"]
)

test_paths = set(
    test_df["filepath"]
)


train_val_overlap = (
    train_paths &
    validation_paths
)

train_test_overlap = (
    train_paths &
    test_paths
)

validation_test_overlap = (
    validation_paths &
    test_paths
)


print("\n==========================================")
print("OVERLAP CHECK")
print("==========================================")

print(
    f"\nTrain - Validation overlap: "
    f"{len(train_val_overlap)}"
)

print(
    f"Train - Test overlap: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation - Test overlap: "
    f"{len(validation_test_overlap)}"
)


# ============================================================
# VERIFY TOTAL
# ============================================================

all_paths = (
    train_paths |
    validation_paths |
    test_paths
)

expected_total = len(train_df) + len(test_df)

print(
    f"\nUnique images across all splits: "
    f"{len(all_paths):,}"
)

print(
    f"Expected images: "
    f"{expected_total:,}"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

if (
    len(train_val_overlap) == 0
    and
    len(train_test_overlap) == 0
    and
    len(validation_test_overlap) == 0
    and
    len(all_paths) == expected_total
):

    print("\n SPLIT VERIFICATION PASSED")

else:

    print("\n SPLIT VERIFICATION FAILED")

    raise RuntimeError(
        "Image overlap or count mismatch detected."
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n==========================================")
print("MANIFESTS SAVED")
print("==========================================")

print(
    f"\nTraining:\n{train_file}"
)

print(
    f"\nValidation:\n{validation_file}"
)

print(
    f"\nTest:\n{test_file}"
)


print("\n==========================================")
print("PHASE 3.1 COMPLETE")
print("==========================================")