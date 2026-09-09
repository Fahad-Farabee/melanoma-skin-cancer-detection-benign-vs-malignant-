import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PHASE 2 - EDA 2: VISUAL EXPLORATION
# ============================================================

# Dataset path
DATASET_ROOT = r"D:/university/summer 26/Data Mining/research project/archive"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")

# Output directory
OUTPUT_DIR = os.path.join(DATASET_ROOT, "phase2_eda")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "sample_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Reproducible random selection
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Number of images per class
N_SAMPLES = 12

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------
# Get image files
# ------------------------------------------------------------

def get_image_files(folder):
    files = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if (
            os.path.isfile(filepath)
            and os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ):
            files.append(filename)

    return sorted(files)


benign_dir = os.path.join(TRAIN_DIR, "Benign")
malignant_dir = os.path.join(TRAIN_DIR, "Malignant")

benign_files = get_image_files(benign_dir)
malignant_files = get_image_files(malignant_dir)


# ------------------------------------------------------------
# Randomly select images
# ------------------------------------------------------------

benign_samples = random.sample(benign_files, N_SAMPLES)
malignant_samples = random.sample(malignant_files, N_SAMPLES)


# ------------------------------------------------------------
# Save selected filenames
# ------------------------------------------------------------

selection_data = []

for filename in benign_samples:
    selection_data.append([
        "Train",
        "Benign",
        filename
    ])

for filename in malignant_samples:
    selection_data.append([
        "Train",
        "Malignant",
        filename
    ])

selection_df = pd.DataFrame(
    selection_data,
    columns=["Split", "Class", "Filename"]
)

selection_csv = os.path.join(
    OUTPUT_DIR,
    "eda2_selected_images.csv"
)

selection_df.to_csv(selection_csv, index=False)


# ------------------------------------------------------------
# Function to create image grid
# ------------------------------------------------------------

def create_grid(files, folder, class_name, output_filename):

    rows = 3
    cols = 4

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 10)
    )

    axes = axes.flatten()

    for i, filename in enumerate(files):

        filepath = os.path.join(folder, filename)

        try:
            image = Image.open(filepath).convert("RGB")

            axes[i].imshow(image)
            axes[i].set_title(
                f"{class_name}\n{filename}",
                fontsize=8
            )
            axes[i].axis("off")

        except Exception as e:

            axes[i].text(
                0.5,
                0.5,
                "Image Error",
                ha="center",
                va="center"
            )

            axes[i].axis("off")

            print(
                f"Error loading {filepath}: {e}"
            )

    plt.suptitle(
        f"EDA-2: Random Training Images - {class_name}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        SAMPLE_DIR,
        output_filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nSaved: {output_path}")


# ------------------------------------------------------------
# Create Benign grid
# ------------------------------------------------------------

create_grid(
    benign_samples,
    benign_dir,
    "Benign",
    "eda2_benign_samples.png"
)


# ------------------------------------------------------------
# Create Malignant grid
# ------------------------------------------------------------

create_grid(
    malignant_samples,
    malignant_dir,
    "Malignant",
    "eda2_malignant_samples.png"
)


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\n==========================================")
print("EDA-2 COMPLETE")
print("==========================================")

print(f"\nBenign images available: {len(benign_files)}")
print(f"Malignant images available: {len(malignant_files)}")

print(f"\nRandom samples selected: {N_SAMPLES} per class")

print("\nSelected image list:")
print(selection_df.to_string(index=False))

print("\nFiles generated:")
print(selection_csv)
print(os.path.join(SAMPLE_DIR, "eda2_benign_samples.png"))
print(os.path.join(SAMPLE_DIR, "eda2_malignant_samples.png"))