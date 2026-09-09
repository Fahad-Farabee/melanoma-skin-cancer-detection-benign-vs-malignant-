import os
import pandas as pd

# ============================================================
# PHASE 2 - EDA 5
# CONSOLIDATED EDA SUMMARY
# ============================================================

DATASET_ROOT = r"D:/university/summer 26/Data Mining/research project/archive"

EDA_DIR = os.path.join(
    DATASET_ROOT,
    "phase2_eda"
)

OUTLIER_DIR = os.path.join(
    EDA_DIR,
    "eda4_outliers"
)

os.makedirs(EDA_DIR, exist_ok=True)


# ============================================================
# DATASET INFORMATION
# ============================================================

train_benign = 6289
train_malignant = 5590

test_benign = 1000
test_malignant = 1000

train_total = train_benign + train_malignant
test_total = test_benign + test_malignant
total_images = train_total + test_total


# ============================================================
# EDA FINDINGS
# ============================================================

findings = [

    [
        "EDA-1",
        "Class Distribution",
        "Training dataset contains 6,289 Benign images (52.94%) and 5,590 Malignant images (47.06%).",
        "The training dataset is relatively balanced; severe class imbalance is not present."
    ],

    [
        "EDA-1",
        "Test Distribution",
        "Test dataset contains 1,000 Benign and 1,000 Malignant images.",
        "The test dataset is perfectly balanced."
    ],

    [
        "EDA-2",
        "Visual Exploration",
        "Both classes show substantial variation in lesion size, shape, pigmentation, texture, surrounding skin, illumination, and acquisition conditions.",
        "The dataset contains substantial intra-class visual variation."
    ],

    [
        "EDA-2",
        "Class Visual Overlap",
        "Some Benign and Malignant images have visually similar general characteristics.",
        "Classification cannot be assumed to depend on one simple visual feature."
    ],

    [
        "EDA-2",
        "Image Artifacts",
        "Some images contain hairs, measurement markings, circular dermoscopic fields, and varying backgrounds.",
        "Preprocessing and augmentation should account for acquisition-related variation."
    ],

    [
        "EDA-3",
        "Image Dimensions",
        "All analyzed training images are 224×224 pixels with an aspect ratio of 1.0.",
        "Images already have a standardized spatial resolution."
    ],

    [
        "EDA-3",
        "Brightness",
        "Benign images generally have higher mean brightness, while Malignant images are generally shifted toward lower brightness. Considerable overlap remains.",
        "Brightness is a dataset-level characteristic and should not be treated as a standalone diagnostic feature."
    ],

    [
        "EDA-3",
        "Contrast",
        "Malignant images show a broader distribution of pixel-intensity variation than Benign images.",
        "Normalization and augmentation may help the models handle acquisition variation."
    ],

    [
        "EDA-3",
        "RGB Statistics",
        "RGB channel means differ between the two classes, particularly in the Red channel.",
        "Color characteristics may provide useful information, but may also reflect dataset/acquisition bias."
    ],

    [
        "EDA-4",
        "Outlier Analysis",
        "IQR analysis identified statistical outliers in brightness, contrast, and RGB statistics.",
        "Statistical outlier status alone is insufficient justification for removing an image."
    ],

    [
        "EDA-4",
        "Visual Outlier Inspection",
        "Representative extreme images were visually inspected and were found to be valid dermoscopic images rather than clearly corrupted images.",
        "No images should be removed solely because they are statistical outliers."
    ],

    [
        "Phase 1",
        "Exact Duplicate Check",
        "No exact train-test duplicates were identified using exact image hashing.",
        "No evidence of exact train-test image duplication was found."
    ],

    [
        "Phase 1",
        "Near-Duplicate Check",
        "Highly similar pHash candidates were identified; the strongest candidates were manually inspected and no visually identical or near-identical pairs were observed among the inspected samples.",
        "No apparent train-test leakage was identified through the conducted duplicate/near-duplicate checks."
    ]
]


# ============================================================
# CREATE FINDINGS DATAFRAME
# ============================================================

findings_df = pd.DataFrame(
    findings,
    columns=[
        "Phase",
        "EDA_Component",
        "Finding",
        "Research_Implication"
    ]
)


# ============================================================
# SAVE FINDINGS
# ============================================================

findings_file = os.path.join(
    EDA_DIR,
    "eda5_consolidated_findings.csv"
)

findings_df.to_csv(
    findings_file,
    index=False
)


# ============================================================
# CREATE DATASET SUMMARY
# ============================================================

dataset_summary = pd.DataFrame(
    [
        ["Train", "Benign", train_benign, 52.94],
        ["Train", "Malignant", train_malignant, 47.06],
        ["Test", "Benign", test_benign, 50.00],
        ["Test", "Malignant", test_malignant, 50.00]
    ],
    columns=[
        "Split",
        "Class",
        "Image_Count",
        "Percentage"
    ]
)


dataset_summary_file = os.path.join(
    EDA_DIR,
    "eda5_dataset_summary.csv"
)

dataset_summary.to_csv(
    dataset_summary_file,
    index=False
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 2 - EDA 5")
print("CONSOLIDATED EDA SUMMARY")
print("=" * 60)

print("\nDATASET SUMMARY")
print("-" * 60)

print(
    dataset_summary.to_string(index=False)
)

print("\n")
print(f"Training images : {train_total:,}")
print(f"Testing images  : {test_total:,}")
print(f"Total images    : {total_images:,}")

print("\n")
print("=" * 60)
print("KEY EDA FINDINGS")
print("=" * 60)

for i, row in findings_df.iterrows():

    print(
        f"\n{i + 1}. [{row['Phase']}] "
        f"{row['EDA_Component']}"
    )

    print(
        f"   Finding: {row['Finding']}"
    )

    print(
        f"   Implication: {row['Research_Implication']}"
    )


# ============================================================
# FINAL RESEARCH DECISIONS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EDA DECISIONS")
print("=" * 60)

decisions = [
    "1. Keep the complete dataset; no images will be removed solely due to statistical outlier status.",
    "2. Preserve the original test set for final model evaluation.",
    "3. Use the training data for subsequent train/validation splitting.",
    "4. Apply image normalization during preprocessing.",
    "5. Use appropriate augmentation to improve robustness to image acquisition variation.",
    "6. Do not rely on brightness or RGB statistics as standalone classification rules.",
    "7. Monitor potential acquisition/color bias during model evaluation.",
    "8. Report Accuracy together with Precision, Recall/Sensitivity, Specificity, F1-score, and ROC-AUC.",
    "9. Use confusion matrices and error analysis to understand model mistakes.",
    "10. Use Grad-CAM later to investigate whether CNN-based models focus on meaningful lesion regions."
]

for decision in decisions:
    print(decision)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("=" * 60)
print("EDA PHASE COMPLETE")
print("=" * 60)

print("\nFiles created:")

print(
    f"\n{findings_file}"
)

print(
    f"{dataset_summary_file}"
)

print("\n")
print("Phase 2 is now complete.")
print("Next phase: Data Splitting & Preprocessing.")