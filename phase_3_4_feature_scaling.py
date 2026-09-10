import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# PHASE 3.4 - TRADITIONAL ML FEATURE SCALING
# ============================================================

print("=" * 70)
print("PHASE 3.4 - TRADITIONAL ML FEATURE SCALING")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = r"D:\university\summer 26\Data Mining\research project\archive"

TRADITIONAL_ML_DIR = os.path.join(
    BASE_DIR,
    "phase3",
    "traditional_ml"
)

SCALED_DIR = os.path.join(
    TRADITIONAL_ML_DIR,
    "scaled"
)

os.makedirs(SCALED_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(
    TRADITIONAL_ML_DIR,
    "train_features.csv"
)

VAL_FILE = os.path.join(
    TRADITIONAL_ML_DIR,
    "validation_features.csv"
)

TEST_FILE = os.path.join(
    TRADITIONAL_ML_DIR,
    "test_features.csv"
)

SCALER_FILE = os.path.join(
    TRADITIONAL_ML_DIR,
    "scaler.joblib"
)

SUMMARY_FILE = os.path.join(
    TRADITIONAL_ML_DIR,
    "scaling_summary.csv"
)

CONFIG_FILE = os.path.join(
    TRADITIONAL_ML_DIR,
    "scaling_config.txt"
)


# ------------------------------------------------------------
# EXPECTED VALUES
# ------------------------------------------------------------

EXPECTED_FEATURES = 6190

EXPECTED_TRAIN = 9503
EXPECTED_VAL = 2376
EXPECTED_TEST = 2000

METADATA_COLUMNS = [
    "filepath",
    "class",
    "label"
]


# ------------------------------------------------------------
# STEP 1 - VERIFY INPUT FILES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 1: VERIFYING INPUT FEATURE FILES")
print("-" * 70)

input_files = [
    ("TRAIN", TRAIN_FILE, EXPECTED_TRAIN),
    ("VALIDATION", VAL_FILE, EXPECTED_VAL),
    ("TEST", TEST_FILE, EXPECTED_TEST)
]

for name, path, expected_rows in input_files:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} feature file not found:\n{path}"
        )

    print(f"{name}: file found")


# ------------------------------------------------------------
# STEP 2 - LOAD FEATURES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 2: LOADING FEATURE DATA")
print("-" * 70)

start_time = time.time()

print("Loading training features...")
train_df = pd.read_csv(TRAIN_FILE)

print("Loading validation features...")
val_df = pd.read_csv(VAL_FILE)

print("Loading test features...")
test_df = pd.read_csv(TEST_FILE)

print(
    f"\nLoading completed in "
    f"{time.time() - start_time:.1f} seconds"
)


# ------------------------------------------------------------
# STEP 3 - VERIFY STRUCTURE
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 3: VERIFYING FEATURE STRUCTURE")
print("-" * 70)


def verify_dataframe(df, name, expected_rows):

    print(f"\n{name}")

    # Row count
    rows = len(df)

    print(f"Rows: {rows}")

    if rows != expected_rows:
        raise ValueError(
            f"{name}: expected {expected_rows} rows, "
            f"found {rows}"
        )

    # Metadata columns
    for column in METADATA_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"{name}: missing required column '{column}'"
            )

    # Feature columns
    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    feature_count = len(feature_columns)

    print(f"Feature columns: {feature_count}")

    if feature_count != EXPECTED_FEATURES:
        raise ValueError(
            f"{name}: expected {EXPECTED_FEATURES} features, "
            f"found {feature_count}"
        )

    # Label check
    unique_labels = sorted(
        df["label"].dropna().unique().tolist()
    )

    print(f"Labels: {unique_labels}")

    if unique_labels != [0, 1]:
        raise ValueError(
            f"{name}: labels must be [0, 1]"
        )

    # Class-label consistency
    class_label_errors = (
        ((df["class"] == "Benign") & (df["label"] != 0))
        |
        ((df["class"] == "Malignant") & (df["label"] != 1))
    ).sum()

    print(f"Class-label errors: {class_label_errors}")

    if class_label_errors != 0:
        raise ValueError(
            f"{name}: class-label mismatch detected"
        )

    return feature_columns


train_feature_columns = verify_dataframe(
    train_df,
    "TRAIN",
    EXPECTED_TRAIN
)

val_feature_columns = verify_dataframe(
    val_df,
    "VALIDATION",
    EXPECTED_VAL
)

test_feature_columns = verify_dataframe(
    test_df,
    "TEST",
    EXPECTED_TEST
)


# ------------------------------------------------------------
# STEP 4 - VERIFY FEATURE COLUMN CONSISTENCY
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 4: VERIFYING FEATURE COLUMN CONSISTENCY")
print("-" * 70)

if train_feature_columns != val_feature_columns:
    raise ValueError(
        "Training and validation feature columns do not match."
    )

if train_feature_columns != test_feature_columns:
    raise ValueError(
        "Training and test feature columns do not match."
    )

print("Training vs Validation: MATCH")
print("Training vs Test: MATCH")
print("Feature ordering: CONSISTENT")


# ------------------------------------------------------------
# STEP 5 - EXTRACT NUMERICAL MATRICES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 5: PREPARING NUMERICAL FEATURE MATRICES")
print("-" * 70)

X_train = train_df[train_feature_columns].to_numpy(
    dtype=np.float32
)

X_val = val_df[val_feature_columns].to_numpy(
    dtype=np.float32
)

X_test = test_df[test_feature_columns].to_numpy(
    dtype=np.float32
)

y_train = train_df["label"].to_numpy(
    dtype=np.int64
)

y_val = val_df["label"].to_numpy(
    dtype=np.int64
)

y_test = test_df["label"].to_numpy(
    dtype=np.int64
)

print(f"Training matrix:   {X_train.shape}")
print(f"Validation matrix: {X_val.shape}")
print(f"Test matrix:       {X_test.shape}")


# ------------------------------------------------------------
# STEP 6 - CHECK NaN / INF
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 6: CHECKING FOR NaN / INF VALUES")
print("-" * 70)

for name, X in [
    ("TRAIN", X_train),
    ("VALIDATION", X_val),
    ("TEST", X_test)
]:

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    print(
        f"{name}: NaN={nan_count} | INF={inf_count}"
    )

    if nan_count != 0 or inf_count != 0:
        raise ValueError(
            f"{name}: NaN or Inf values detected."
        )

print("Numerical feature quality check: PASS")


# ------------------------------------------------------------
# STEP 7 - FIT STANDARD SCALER ON TRAINING ONLY
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 7: FITTING STANDARD SCALER")
print("-" * 70)

print("IMPORTANT:")
print("Scaler will be FIT ONLY on training data.")
print("Validation and test data will NOT be used for fitting.")

scaler = StandardScaler()

scaler.fit(X_train)

print("Scaler fitting completed.")


# ------------------------------------------------------------
# STEP 8 - TRANSFORM ALL SPLITS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 8: TRANSFORMING FEATURES")
print("-" * 70)

start_time = time.time()

X_train_scaled = scaler.transform(X_train).astype(
    np.float32
)

X_val_scaled = scaler.transform(X_val).astype(
    np.float32
)

X_test_scaled = scaler.transform(X_test).astype(
    np.float32
)

print(
    f"Transformation completed in "
    f"{time.time() - start_time:.1f} seconds"
)

print(f"Scaled training matrix:   {X_train_scaled.shape}")
print(f"Scaled validation matrix: {X_val_scaled.shape}")
print(f"Scaled test matrix:       {X_test_scaled.shape}")


# ------------------------------------------------------------
# STEP 9 - VERIFY TRAINING STANDARDIZATION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 9: VERIFYING STANDARDIZATION")
print("-" * 70)

train_means = X_train_scaled.mean(axis=0)
train_stds = X_train_scaled.std(axis=0)

mean_abs_max = np.max(np.abs(train_means))
std_deviation_max = np.max(np.abs(train_stds - 1))

print(
    f"Maximum absolute training feature mean: "
    f"{mean_abs_max:.8f}"
)

print(
    f"Maximum deviation of training feature std from 1: "
    f"{std_deviation_max:.8f}"
)

if mean_abs_max > 1e-4:
    print("WARNING: Some training feature means differ from 0.")

if std_deviation_max > 1e-4:
    print("WARNING: Some training feature std values differ from 1.")

print("Training standardization check completed.")


# ------------------------------------------------------------
# STEP 10 - SAVE SCALED MATRICES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 10: SAVING SCALED FEATURES")
print("-" * 70)

np.savez_compressed(
    os.path.join(
        SCALED_DIR,
        "train_X_scaled.npz"
    ),
    X=X_train_scaled,
    y=y_train
)

np.savez_compressed(
    os.path.join(
        SCALED_DIR,
        "validation_X_scaled.npz"
    ),
    X=X_val_scaled,
    y=y_val
)

np.savez_compressed(
    os.path.join(
        SCALED_DIR,
        "test_X_scaled.npz"
    ),
    X=X_test_scaled,
    y=y_test
)

print("Saved:")
print("  train_X_scaled.npz")
print("  validation_X_scaled.npz")
print("  test_X_scaled.npz")


# ------------------------------------------------------------
# STEP 11 - SAVE SCALER
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 11: SAVING SCALER")
print("-" * 70)

joblib.dump(
    scaler,
    SCALER_FILE
)

print(f"Scaler saved to:")
print(SCALER_FILE)


# ------------------------------------------------------------
# STEP 12 - SAVE SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame([
    {
        "split": "train",
        "samples": X_train.shape[0],
        "features": X_train.shape[1],
        "benign": int((y_train == 0).sum()),
        "malignant": int((y_train == 1).sum()),
        "scaling": "StandardScaler",
        "scaler_fitted_on": "train"
    },
    {
        "split": "validation",
        "samples": X_val.shape[0],
        "features": X_val.shape[1],
        "benign": int((y_val == 0).sum()),
        "malignant": int((y_val == 1).sum()),
        "scaling": "StandardScaler",
        "scaler_fitted_on": "train"
    },
    {
        "split": "test",
        "samples": X_test.shape[0],
        "features": X_test.shape[1],
        "benign": int((y_test == 0).sum()),
        "malignant": int((y_test == 1).sum()),
        "scaling": "StandardScaler",
        "scaler_fitted_on": "train"
    }
])

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print(f"\nSummary saved to:")
print(SUMMARY_FILE)


# ------------------------------------------------------------
# STEP 13 - SAVE CONFIGURATION
# ------------------------------------------------------------

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 3.4 - FEATURE SCALING CONFIGURATION\n"
        "=========================================\n\n"
        "Feature source:\n"
        "Phase 3.3 RGB + LBP + HOG features\n\n"
        "Total features: 6190\n\n"
        "Scaling method:\n"
        "StandardScaler\n\n"
        "Formula:\n"
        "z = (x - mean) / standard_deviation\n\n"
        "Data leakage prevention:\n"
        "Scaler fitted ONLY on training features.\n"
        "Validation and test features were transformed\n"
        "using the same training-fitted scaler.\n\n"
        "Training samples: 9503\n"
        "Validation samples: 2376\n"
        "Test samples: 2000\n\n"
        "Feature groups:\n"
        "RGB Histogram: 96\n"
        "LBP Texture: 10\n"
        "HOG: 6084\n"
        "Combined: 6190\n"
    )

print(f"Configuration saved to:")
print(CONFIG_FILE)


# ------------------------------------------------------------
# FINAL VERIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 3.4 SCALING COMPLETED")
print("=" * 70)

print("\nFinal matrices:")
print(f"  Train:      {X_train_scaled.shape}")
print(f"  Validation: {X_val_scaled.shape}")
print(f"  Test:       {X_test_scaled.shape}")

print("\nScaling method:")
print("  StandardScaler")

print("\nLeakage prevention:")
print("  [OK] Scaler fitted only on training data")
print("  [OK] Validation transformed using training scaler")
print("  [OK] Test transformed using training scaler")

print("\nGenerated files:")
print("  [OK] train_X_scaled.npz")
print("  [OK] validation_X_scaled.npz")
print("  [OK] test_X_scaled.npz")
print("  [OK] scaler.joblib")
print("  [OK] scaling_summary.csv")
print("  [OK] scaling_config.txt")

print("\n" + "=" * 70)