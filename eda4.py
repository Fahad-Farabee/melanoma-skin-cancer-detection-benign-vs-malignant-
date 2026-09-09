import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PHASE 2 - EDA 4: OUTLIER & IMAGE-QUALITY ANALYSIS
# ============================================================

DATASET_ROOT = r"D:/university/summer 26/Data Mining/research project/archive"

EDA_DIR = os.path.join(DATASET_ROOT, "phase2_eda")

STATS_FILE = os.path.join(
    EDA_DIR,
    "eda3_image_statistics.csv"
)

OUTLIER_DIR = os.path.join(
    EDA_DIR,
    "eda4_outliers"
)

os.makedirs(OUTLIER_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# LOAD EDA-3 STATISTICS
# ============================================================

print("\n==========================================")
print("PHASE 2 - EDA 4")
print("OUTLIER & IMAGE-QUALITY ANALYSIS")
print("==========================================\n")

if not os.path.exists(STATS_FILE):
    raise FileNotFoundError(
        f"Could not find:\n{STATS_FILE}\n"
        "Please run EDA-3 first."
    )

df = pd.read_csv(STATS_FILE)

print(f"Loaded statistics for {len(df):,} images.")


# ============================================================
# IQR OUTLIER FUNCTION
# ============================================================

def calculate_iqr_bounds(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return q1, q3, iqr, lower, upper


# ============================================================
# VARIABLES TO ANALYZE
# ============================================================

variables = [
    "Brightness_Mean",
    "Brightness_Std",
    "Red_Mean",
    "Green_Mean",
    "Blue_Mean"
]


# ============================================================
# CALCULATE OUTLIERS SEPARATELY BY CLASS
# ============================================================

all_outliers = []
summary_records = []

for class_name in ["Benign", "Malignant"]:

    class_df = df[df["Class"] == class_name].copy()

    print("\n------------------------------------------")
    print(f"Class: {class_name}")
    print("------------------------------------------")

    for variable in variables:

        q1, q3, iqr, lower, upper = calculate_iqr_bounds(
            class_df[variable]
        )

        mask = (
            (class_df[variable] < lower) |
            (class_df[variable] > upper)
        )

        outliers = class_df[mask].copy()

        # Record summary
        summary_records.append([
            class_name,
            variable,
            q1,
            q3,
            iqr,
            lower,
            upper,
            len(outliers),
            len(outliers) / len(class_df) * 100
        ])

        print(
            f"\n{variable}"
        )
        print(
            f"  Q1: {q1:.2f}"
        )
        print(
            f"  Q3: {q3:.2f}"
        )
        print(
            f"  IQR: {iqr:.2f}"
        )
        print(
            f"  Lower bound: {lower:.2f}"
        )
        print(
            f"  Upper bound: {upper:.2f}"
        )
        print(
            f"  Outliers: {len(outliers):,}"
            f" ({len(outliers)/len(class_df)*100:.2f}%)"
        )

        # Store outliers
        for _, row in outliers.iterrows():

            all_outliers.append({
                "Class": class_name,
                "Filename": row["Filename"],
                "Variable": variable,
                "Value": row[variable],
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower_Bound": lower,
                "Upper_Bound": upper
            })


# ============================================================
# SAVE OUTLIER SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_records,
    columns=[
        "Class",
        "Variable",
        "Q1",
        "Q3",
        "IQR",
        "Lower_Bound",
        "Upper_Bound",
        "Outlier_Count",
        "Outlier_Percentage"
    ]
)

summary_file = os.path.join(
    OUTLIER_DIR,
    "eda4_outlier_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n==========================================")
print("OUTLIER SUMMARY SAVED")
print("==========================================")

print(summary_df.to_string(index=False))

print(f"\nSaved to:\n{summary_file}")


# ============================================================
# SAVE ALL OUTLIERS
# ============================================================

outliers_df = pd.DataFrame(all_outliers)

outlier_file = os.path.join(
    OUTLIER_DIR,
    "eda4_all_outliers.csv"
)

outliers_df.to_csv(
    outlier_file,
    index=False
)

print(f"\nAll outliers saved to:\n{outlier_file}")


# ============================================================
# CREATE UNIQUE OUTLIER IMAGE LIST
# ============================================================

unique_outliers = (
    outliers_df[
        ["Class", "Filename"]
    ]
    .drop_duplicates()
)

unique_outlier_file = os.path.join(
    OUTLIER_DIR,
    "eda4_unique_outlier_images.csv"
)

unique_outliers.to_csv(
    unique_outlier_file,
    index=False
)

print(
    f"\nUnique outlier image list saved to:\n"
    f"{unique_outlier_file}"
)


# ============================================================
# FIND MOST EXTREME IMAGES
# ============================================================

extreme_records = []

for class_name in ["Benign", "Malignant"]:

    class_df = df[
        df["Class"] == class_name
    ].copy()

    # --------------------------------------------------------
    # Darkest
    # --------------------------------------------------------

    darkest = class_df.nsmallest(
        5,
        "Brightness_Mean"
    )

    for _, row in darkest.iterrows():

        extreme_records.append([
            class_name,
            "Darkest",
            row["Filename"],
            row["Brightness_Mean"]
        ])

    # --------------------------------------------------------
    # Brightest
    # --------------------------------------------------------

    brightest = class_df.nlargest(
        5,
        "Brightness_Mean"
    )

    for _, row in brightest.iterrows():

        extreme_records.append([
            class_name,
            "Brightest",
            row["Filename"],
            row["Brightness_Mean"]
        ])

    # --------------------------------------------------------
    # Lowest contrast
    # --------------------------------------------------------

    lowest_contrast = class_df.nsmallest(
        5,
        "Brightness_Std"
    )

    for _, row in lowest_contrast.iterrows():

        extreme_records.append([
            class_name,
            "Lowest_Contrast",
            row["Filename"],
            row["Brightness_Std"]
        ])

    # --------------------------------------------------------
    # Highest contrast
    # --------------------------------------------------------

    highest_contrast = class_df.nlargest(
        5,
        "Brightness_Std"
    )

    for _, row in highest_contrast.iterrows():

        extreme_records.append([
            class_name,
            "Highest_Contrast",
            row["Filename"],
            row["Brightness_Std"]
        ])


extreme_df = pd.DataFrame(
    extreme_records,
    columns=[
        "Class",
        "Category",
        "Filename",
        "Value"
    ]
)

extreme_file = os.path.join(
    OUTLIER_DIR,
    "eda4_extreme_images.csv"
)

extreme_df.to_csv(
    extreme_file,
    index=False
)

print(
    f"\nExtreme image list saved to:\n"
    f"{extreme_file}"
)


# ============================================================
# IMAGE DIRECTORY HELPER
# ============================================================

def get_image_path(class_name, filename):

    return os.path.join(
        DATASET_ROOT,
        "train",
        class_name,
        filename
    )


# ============================================================
# CREATE VISUAL GRID
# ============================================================

def create_extreme_grid(
    class_name,
    category,
    n=5
):

    subset = extreme_df[
        (extreme_df["Class"] == class_name) &
        (extreme_df["Category"] == category)
    ].head(n)

    if len(subset) == 0:
        return

    fig, axes = plt.subplots(
        1,
        len(subset),
        figsize=(15, 4)
    )

    if len(subset) == 1:
        axes = [axes]

    for ax, (_, row) in zip(
        axes,
        subset.iterrows()
    ):

        filepath = get_image_path(
            class_name,
            row["Filename"]
        )

        try:

            image = Image.open(
                filepath
            ).convert("RGB")

            ax.imshow(image)

            ax.set_title(
                f"{row['Filename']}\n"
                f"{row['Value']:.2f}",
                fontsize=8
            )

            ax.axis("off")

        except Exception as e:

            ax.text(
                0.5,
                0.5,
                "Image Error",
                ha="center",
                va="center"
            )

            ax.axis("off")

            print(
                f"Error loading {filepath}: {e}"
            )

    plt.suptitle(
        f"EDA-4: {class_name} - {category}",
        fontsize=14
    )

    plt.tight_layout()

    filename = (
        f"eda4_{class_name.lower()}_"
        f"{category.lower()}.png"
    )

    output_path = os.path.join(
        OUTLIER_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved: {output_path}")


# ============================================================
# GENERATE EXTREME IMAGE GRIDS
# ============================================================

for class_name in ["Benign", "Malignant"]:

    for category in [
        "Darkest",
        "Brightest",
        "Lowest_Contrast",
        "Highest_Contrast"
    ]:

        create_extreme_grid(
            class_name,
            category
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("EDA-4 COMPLETE")
print("==========================================")

print(
    f"\nTotal images analyzed: {len(df):,}"
)

print(
    f"Unique images identified as statistical "
    f"outliers: {len(unique_outliers):,}"
)

print(
    "\nIMPORTANT:"
)

print(
    "Statistical outliers are NOT automatically "
    "bad images."
)

print(
    "They must be visually inspected before "
    "considering any removal."
)

print(
    "\nOutput directory:"
)

print(OUTLIER_DIR)