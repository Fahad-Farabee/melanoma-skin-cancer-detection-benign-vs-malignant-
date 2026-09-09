import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PHASE 2 - EDA 1: CLASS DISTRIBUTION
# ============================================================

# Dataset path
DATASET_ROOT = r"D:/university/summer 26/Data Mining/research project/archive"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# Output directory
OUTPUT_DIR = os.path.join(DATASET_ROOT, "phase2_eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Count images
# ------------------------------------------------------------

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in extensions
    ])


train_benign = count_images(os.path.join(TRAIN_DIR, "Benign"))
train_malignant = count_images(os.path.join(TRAIN_DIR, "Malignant"))

test_benign = count_images(os.path.join(TEST_DIR, "Benign"))
test_malignant = count_images(os.path.join(TEST_DIR, "Malignant"))


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

data = [
    ["Train", "Benign", train_benign],
    ["Train", "Malignant", train_malignant],
    ["Test", "Benign", test_benign],
    ["Test", "Malignant", test_malignant],
]

df = pd.DataFrame(
    data,
    columns=["Split", "Class", "Image_Count"]
)


# ------------------------------------------------------------
# Calculate percentages within each split
# ------------------------------------------------------------

df["Percentage"] = (
    df.groupby("Split")["Image_Count"]
      .transform(lambda x: x / x.sum() * 100)
)

df["Percentage"] = df["Percentage"].round(2)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n==========================================")
print("PHASE 2 - EDA 1: CLASS DISTRIBUTION")
print("==========================================\n")

print(df.to_string(index=False))


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

csv_path = os.path.join(
    OUTPUT_DIR,
    "eda1_class_distribution.csv"
)

df.to_csv(csv_path, index=False)

print("\nCSV saved to:")
print(csv_path)


# ------------------------------------------------------------
# Create chart
# ------------------------------------------------------------

# Training data
train_df = df[df["Split"] == "Train"]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    train_df["Class"],
    train_df["Image_Count"]
)

plt.title("Training Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

# Add values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR,
    "eda1_training_class_distribution.png"
)

plt.savefig(chart_path, dpi=300)
plt.show()

print("\nChart saved to:")
print(chart_path)


# ------------------------------------------------------------
# Test distribution chart
# ------------------------------------------------------------

test_df = df[df["Split"] == "Test"]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    test_df["Class"],
    test_df["Image_Count"]
)

plt.title("Test Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR,
    "eda1_test_class_distribution.png"
)

plt.savefig(chart_path, dpi=300)
plt.show()

print("\nChart saved to:")
print(chart_path)


print("\n==========================================")
print("EDA-1 COMPLETE")
print("==========================================")