import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PHASE 2 - EDA 3: QUANTITATIVE IMAGE STATISTICS
# ============================================================

DATASET_ROOT = r"D:/university/summer 26/Data Mining/research project/archive"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "phase2_eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            and os.path.splitext(filename)[1].lower()
            in IMAGE_EXTENSIONS
        ):
            files.append(filename)

    return sorted(files)


# ------------------------------------------------------------
# Analyze one image
# ------------------------------------------------------------

def analyze_image(filepath):

    image = Image.open(filepath).convert("RGB")

    img = np.array(image).astype(np.float32)

    height, width, _ = img.shape

    # Aspect ratio
    aspect_ratio = width / height

    # RGB channels
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # Overall brightness
    brightness = np.mean(img)

    # Overall intensity variation / contrast
    contrast = np.std(img)

    return {
        "Width": width,
        "Height": height,
        "Aspect_Ratio": aspect_ratio,

        "Brightness_Mean": brightness,
        "Brightness_Std": contrast,

        "Red_Mean": np.mean(red),
        "Green_Mean": np.mean(green),
        "Blue_Mean": np.mean(blue),

        "Red_Std": np.std(red),
        "Green_Std": np.std(green),
        "Blue_Std": np.std(blue)
    }


# ------------------------------------------------------------
# Process training dataset
# ------------------------------------------------------------

records = []

classes = ["Benign", "Malignant"]

print("\n==========================================")
print("PHASE 2 - EDA 3")
print("QUANTITATIVE IMAGE STATISTICS")
print("==========================================\n")

for class_name in classes:

    class_dir = os.path.join(TRAIN_DIR, class_name)

    files = get_image_files(class_dir)

    print(f"Processing {class_name}: {len(files)} images")

    for i, filename in enumerate(files):

        filepath = os.path.join(class_dir, filename)

        try:

            stats = analyze_image(filepath)

            stats["Class"] = class_name
            stats["Filename"] = filename

            records.append(stats)

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(
                f"  Processed {i + 1}/{len(files)}"
            )


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

df = pd.DataFrame(records)

# Reorder columns
df = df[
    [
        "Class",
        "Filename",
        "Width",
        "Height",
        "Aspect_Ratio",
        "Brightness_Mean",
        "Brightness_Std",
        "Red_Mean",
        "Green_Mean",
        "Blue_Mean",
        "Red_Std",
        "Green_Std",
        "Blue_Std"
    ]
]


# ------------------------------------------------------------
# Save per-image statistics
# ------------------------------------------------------------

csv_path = os.path.join(
    OUTPUT_DIR,
    "eda3_image_statistics.csv"
)

df.to_csv(csv_path, index=False)

print("\nPer-image statistics saved to:")
print(csv_path)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

numeric_columns = [
    "Width",
    "Height",
    "Aspect_Ratio",
    "Brightness_Mean",
    "Brightness_Std",
    "Red_Mean",
    "Green_Mean",
    "Blue_Mean",
    "Red_Std",
    "Green_Std",
    "Blue_Std"
]

summary = (
    df.groupby("Class")[numeric_columns]
      .agg(["mean", "std", "min", "max"])
)


summary_csv = os.path.join(
    OUTPUT_DIR,
    "eda3_summary_statistics.csv"
)

summary.to_csv(summary_csv)

print("\nSummary statistics saved to:")
print(summary_csv)


# ============================================================
# PRINT IMPORTANT RESULTS
# ============================================================

print("\n==========================================")
print("SUMMARY")
print("==========================================\n")

print(
    df.groupby("Class")[
        [
            "Width",
            "Height",
            "Aspect_Ratio",
            "Brightness_Mean",
            "Brightness_Std",
            "Red_Mean",
            "Green_Mean",
            "Blue_Mean"
        ]
    ].mean().round(2)
)


# ============================================================
# GRAPH 1: BRIGHTNESS
# ============================================================

plt.figure(figsize=(9, 6))

for class_name in classes:

    subset = df[df["Class"] == class_name]

    plt.hist(
        subset["Brightness_Mean"],
        bins=40,
        alpha=0.6,
        label=class_name
    )

plt.title("Mean Brightness Distribution")
plt.xlabel("Mean Pixel Intensity")
plt.ylabel("Number of Images")
plt.legend()

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "eda3_brightness_distribution.png"
)

plt.savefig(path, dpi=300)
plt.show()

print(f"Saved: {path}")


# ============================================================
# GRAPH 2: CONTRAST
# ============================================================

plt.figure(figsize=(9, 6))

for class_name in classes:

    subset = df[df["Class"] == class_name]

    plt.hist(
        subset["Brightness_Std"],
        bins=40,
        alpha=0.6,
        label=class_name
    )

plt.title("Image Contrast / Intensity Variation")
plt.xlabel("Pixel Intensity Standard Deviation")
plt.ylabel("Number of Images")
plt.legend()

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "eda3_contrast_distribution.png"
)

plt.savefig(path, dpi=300)
plt.show()

print(f"Saved: {path}")


# ============================================================
# GRAPH 3: RGB MEAN DISTRIBUTIONS
# ============================================================

plt.figure(figsize=(10, 6))

for class_name in classes:

    subset = df[df["Class"] == class_name]

    plt.hist(
        subset["Red_Mean"],
        bins=40,
        alpha=0.5,
        label=f"{class_name} - Red"
    )

    plt.hist(
        subset["Green_Mean"],
        bins=40,
        alpha=0.5,
        label=f"{class_name} - Green"
    )

    plt.hist(
        subset["Blue_Mean"],
        bins=40,
        alpha=0.5,
        label=f"{class_name} - Blue"
    )

plt.title("RGB Channel Mean Distributions")
plt.xlabel("Mean Pixel Intensity")
plt.ylabel("Number of Images")
plt.legend(fontsize=8)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "eda3_rgb_distribution.png"
)

plt.savefig(path, dpi=300)
plt.show()

print(f"Saved: {path}")


# ============================================================
# GRAPH 4: ASPECT RATIO
# ============================================================

plt.figure(figsize=(9, 6))

for class_name in classes:

    subset = df[df["Class"] == class_name]

    plt.hist(
        subset["Aspect_Ratio"],
        bins=20,
        alpha=0.6,
        label=class_name
    )

plt.title("Image Aspect Ratio Distribution")
plt.xlabel("Width / Height")
plt.ylabel("Number of Images")
plt.legend()

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "eda3_aspect_ratio_distribution.png"
)

plt.savefig(path, dpi=300)
plt.show()

print(f"Saved: {path}")


# ============================================================
# COMPLETION
# ============================================================

print("\n==========================================")
print("EDA-3 COMPLETE")
print("==========================================")

print(f"\nTotal images analyzed: {len(df)}")

print("\nImages by class:")
print(df["Class"].value_counts())

print("\nOutput directory:")
print(OUTPUT_DIR)