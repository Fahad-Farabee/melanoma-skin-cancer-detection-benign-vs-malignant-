from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

from skimage.feature import local_binary_pattern, hog


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:/university/summer 26/Data Mining/research project"
)

ARCHIVE_DIR = PROJECT_ROOT / "archive"

PHASE3_DIR = ARCHIVE_DIR / "phase3"

OUTPUT_DIR = PHASE3_DIR / "traditional_ml"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_MANIFEST = (
    PHASE3_DIR / "train_manifest.csv"
)

VAL_MANIFEST = (
    PHASE3_DIR / "validation_manifest.csv"
)

TEST_MANIFEST = (
    PHASE3_DIR / "test_manifest.csv"
)


# ============================================================
# 2. EXPECTED SPLIT COUNTS
# ============================================================

EXPECTED_COUNTS = {

    "train": 9503,

    "validation": 2376,

    "test": 2000

}


EXPECTED_CLASS_COUNTS = {

    "train": {
        "Benign": 5031,
        "Malignant": 4472
    },

    "validation": {
        "Benign": 1258,
        "Malignant": 1118
    },

    "test": {
        "Benign": 1000,
        "Malignant": 1000
    }

}


EXPECTED_LABEL_MAPPING = {

    "Benign": 0,

    "Malignant": 1

}


# ============================================================
# 3. FEATURE CONFIGURATION
# ============================================================

# -----------------------------
# RGB Histogram
# -----------------------------

RGB_BINS = 32

RGB_RANGE = (
    0,
    256
)

RGB_FEATURE_COUNT = (
    RGB_BINS * 3
)


# -----------------------------
# LBP
# -----------------------------

LBP_POINTS = 8

LBP_RADIUS = 1

LBP_METHOD = "uniform"

# For uniform LBP with P=8,
# there are P+2 = 10 histogram bins.

LBP_FEATURE_COUNT = (
    LBP_POINTS + 2
)


# -----------------------------
# HOG
# -----------------------------

HOG_ORIENTATIONS = 9

HOG_PIXELS_PER_CELL = (
    16,
    16
)

HOG_CELLS_PER_BLOCK = (
    2,
    2
)

HOG_BLOCK_NORM = "L2-Hys"


# ============================================================
# 4. OUTPUT FILES
# ============================================================

TRAIN_FEATURES_FILE = (
    OUTPUT_DIR / "train_features.csv"
)

VAL_FEATURES_FILE = (
    OUTPUT_DIR / "validation_features.csv"
)

TEST_FEATURES_FILE = (
    OUTPUT_DIR / "test_features.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "feature_extraction_summary.csv"
)

CONFIG_FILE = (
    OUTPUT_DIR / "feature_extraction_config.txt"
)


# ============================================================
# 5. IMAGE SETTINGS
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# 6. RGB COLOR HISTOGRAM
# ============================================================

def extract_rgb_histogram(image):
    """
    Extract normalized RGB histograms.

    32 bins for Red
    32 bins for Green
    32 bins for Blue

    Total = 96 features.
    """

    image_array = np.asarray(
        image
    )

    features = []

    for channel in range(3):

        histogram, _ = np.histogram(

            image_array[:, :, channel],

            bins=RGB_BINS,

            range=RGB_RANGE,

            density=False

        )

        # Convert counts to proportions.
        histogram = (
            histogram.astype(np.float32)
            / histogram.sum()
        )

        features.extend(
            histogram.tolist()
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# 7. LBP TEXTURE FEATURES
# ============================================================

def extract_lbp_features(image):
    """
    Extract Local Binary Pattern texture histogram.

    P = 8 neighboring points
    Radius = 1
    Method = uniform

    Uniform LBP produces 10 histogram bins.
    """

    image_array = np.asarray(
        image
    ).astype(
        np.float32
    )

    # Convert RGB to grayscale.

    grayscale = (

        0.299 * image_array[:, :, 0]

        + 0.587 * image_array[:, :, 1]

        + 0.114 * image_array[:, :, 2]

    )

    lbp = local_binary_pattern(

        grayscale,

        P=LBP_POINTS,

        R=LBP_RADIUS,

        method=LBP_METHOD

    )

    histogram, _ = np.histogram(

        lbp.ravel(),

        bins=np.arange(
            0,
            LBP_POINTS + 3
        ),

        range=(
            0,
            LBP_POINTS + 2
        )

    )

    histogram = (

        histogram.astype(
            np.float32
        )

        / histogram.sum()

    )

    return histogram


# ============================================================
# 8. HOG SHAPE / EDGE FEATURES
# ============================================================

def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients.

    HOG captures local edge directions and
    shape/structure information.
    """

    image_array = np.asarray(
        image
    )

    grayscale = (

        0.299 * image_array[:, :, 0]

        + 0.587 * image_array[:, :, 1]

        + 0.114 * image_array[:, :, 2]

    )

    features = hog(

        grayscale,

        orientations=HOG_ORIENTATIONS,

        pixels_per_cell=HOG_PIXELS_PER_CELL,

        cells_per_block=HOG_CELLS_PER_BLOCK,

        block_norm=HOG_BLOCK_NORM,

        feature_vector=True

    )

    return np.asarray(

        features,

        dtype=np.float32

    )


# ============================================================
# 9. EXTRACT ALL FEATURES FROM ONE IMAGE
# ============================================================

def extract_features(image_path):
    """
    Extract all feature groups from one image.

    Returns:
        RGB features
        LBP features
        HOG features
        Combined features
    """

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        # Dataset images are already 224x224.
        # Resize is included as a safety guarantee.

        if image.size != (
            224,
            224
        ):

            image = image.resize(
                (224, 224)
            )

        rgb_features = (
            extract_rgb_histogram(
                image
            )
        )

        lbp_features = (
            extract_lbp_features(
                image
            )
        )

        hog_features = (
            extract_hog_features(
                image
            )
        )

    combined_features = np.concatenate(

        [
            rgb_features,
            lbp_features,
            hog_features
        ]

    )

    return (

        rgb_features,

        lbp_features,

        hog_features,

        combined_features

    )


# ============================================================
# 10. VERIFY MANIFEST
# ============================================================

def verify_manifest(
    df,
    split_name
):

    errors = []

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [

        "filepath",
        "class",
        "label"

    ]

    missing_columns = [

        column
        for column in required_columns
        if column not in df.columns

    ]

    if missing_columns:

        errors.append(

            f"Missing columns: "
            f"{missing_columns}"

        )

        return errors


    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    expected_count = (
        EXPECTED_COUNTS[
            split_name
        ]
    )

    if len(df) != expected_count:

        errors.append(

            f"Expected {expected_count} "
            f"rows but found {len(df)}"

        )


    # --------------------------------------------------------
    # Class counts
    # --------------------------------------------------------

    class_counts = (

        df["class"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()

    )


    for class_name, expected_count in (

        EXPECTED_CLASS_COUNTS[
            split_name
        ].items()

    ):

        actual_count = int(

            class_counts.get(
                class_name,
                0
            )

        )

        if actual_count != expected_count:

            errors.append(

                f"{class_name}: expected "
                f"{expected_count}, found "
                f"{actual_count}"

            )


    # --------------------------------------------------------
    # Class / label mapping
    # --------------------------------------------------------

    mapping_errors = 0

    for _, row in df.iterrows():

        class_name = str(
            row["class"]
        ).strip()

        try:

            label = int(
                row["label"]
            )

        except (
            ValueError,
            TypeError
        ):

            mapping_errors += 1

            continue


        expected_label = (
            EXPECTED_LABEL_MAPPING.get(
                class_name
            )
        )


        if expected_label is None:

            mapping_errors += 1

        elif label != expected_label:

            mapping_errors += 1


    if mapping_errors > 0:

        errors.append(

            f"Class/label mapping errors: "
            f"{mapping_errors}"

        )


    # --------------------------------------------------------
    # Image paths
    # --------------------------------------------------------

    missing_images = 0

    for filepath in df["filepath"]:

        path = Path(
            str(filepath)
        )

        if not path.is_absolute():

            path = (
                PROJECT_ROOT / path
            )


        if not path.exists():

            missing_images += 1


    if missing_images > 0:

        errors.append(

            f"Missing image files: "
            f"{missing_images}"

        )


    return errors


# ============================================================
# 11. PROCESS ONE SPLIT
# ============================================================

def process_split(
    df,
    split_name
):

    print("\n" + "=" * 70)

    print(
        f"PROCESSING {split_name.upper()} SPLIT"
    )

    print("=" * 70)


    feature_rows = []

    failed_images = []

    start_time = time.time()


    total = len(df)


    for index, row in df.iterrows():

        image_path = Path(
            str(row["filepath"])
        )


        if not image_path.is_absolute():

            image_path = (
                PROJECT_ROOT / image_path
            )


        try:

            (

                rgb_features,

                lbp_features,

                hog_features,

                combined_features

            ) = extract_features(
                image_path
            )


            record = {

                "filepath":
                    str(image_path),

                "class":
                    str(row["class"]).strip(),

                "label":
                    int(row["label"])

            }


            # -------------------------
            # Color features
            # -------------------------

            for i, value in enumerate(
                rgb_features
            ):

                record[
                    f"rgb_{i+1}"
                ] = float(value)


            # -------------------------
            # LBP features
            # -------------------------

            for i, value in enumerate(
                lbp_features
            ):

                record[
                    f"lbp_{i+1}"
                ] = float(value)


            # -------------------------
            # HOG features
            # -------------------------

            for i, value in enumerate(
                hog_features
            ):

                record[
                    f"hog_{i+1}"
                ] = float(value)


            feature_rows.append(
                record
            )


        except Exception as error:

            failed_images.append({

                "filepath":
                    str(image_path),

                "error":
                    str(error)

            })


        # Progress every 250 images.

        if (
            (index + 1) % 250 == 0
            or
            index + 1 == total
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(

                f"Progress: "
                f"{index + 1}/{total} "
                f"({((index + 1) / total) * 100:.1f}%) "
                f"| Time: {elapsed:.1f}s"

            )


    feature_df = pd.DataFrame(
        feature_rows
    )


    return (
        feature_df,
        failed_images
    )


# ============================================================
# 12. SAVE FEATURE DATA
# ============================================================

def save_features(
    feature_df,
    output_path
):

    feature_df.to_csv(

        output_path,

        index=False,

        float_format="%.8f"

    )

    print(
        f"\nSaved:"
    )

    print(
        output_path
    )

    print(
        f"Rows: {len(feature_df)}"
    )

    print(
        f"Columns: {len(feature_df.columns)}"
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 3.3 - "
        "TRADITIONAL ML FEATURE EXTRACTION"
    )

    print("=" * 70)


    print("\nFeature groups:")

    print(
        f"  RGB Histogram: "
        f"{RGB_FEATURE_COUNT} features"
    )

    print(
        f"  LBP Texture: "
        f"{LBP_FEATURE_COUNT} features"
    )


    # Determine HOG feature count
    # using one example image later.

    print(
        "  HOG: determined automatically"
    )

    print(
        "  Combined: RGB + LBP + HOG"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "  Features are UN SCALED in this phase."
        .replace("UN SCALED", "UNSCALED")
    )

    print(
        "  Standardization will be handled in Phase 3.4."
    )

    print(
        "  No data augmentation is used."
    )


    # ========================================================
    # STEP 1 - LOAD MANIFESTS
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "STEP 1: VERIFYING MANIFESTS"
    )

    print("-" * 70)


    manifests = {

        "train": TRAIN_MANIFEST,

        "validation": VAL_MANIFEST,

        "test": TEST_MANIFEST

    }


    dataframes = {}


    for split_name, manifest_path in (
        manifests.items()
    ):

        if not manifest_path.exists():

            print(
                f"\nERROR: "
                f"{manifest_path} not found."
            )

            sys.exit(1)


        df = pd.read_csv(
            manifest_path
        )


        errors = verify_manifest(

            df,

            split_name

        )


        if errors:

            print(
                f"\n{split_name.upper()}: FAIL"
            )

            for error in errors:

                print(
                    f"  ERROR: {error}"
                )

            sys.exit(1)


        print(

            f"{split_name.upper()}: "
            f"{len(df)} images | "
            f"Benign="
            f"{(df['class'] == 'Benign').sum()} | "
            f"Malignant="
            f"{(df['class'] == 'Malignant').sum()} | "
            f"Status=PASS"

        )


        dataframes[
            split_name
        ] = df


    print(
        "\nAll manifests passed verification."
    )


    # ========================================================
    # STEP 2 - DETERMINE HOG DIMENSION
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "STEP 2: VERIFYING FEATURE DIMENSIONS"
    )

    print("-" * 70)


    first_train_path = Path(

        str(
            dataframes[
                "train"
            ].iloc[0]["filepath"]
        )

    )


    if not first_train_path.is_absolute():

        first_train_path = (
            PROJECT_ROOT
            / first_train_path
        )


    (

        rgb_sample,

        lbp_sample,

        hog_sample,

        combined_sample

    ) = extract_features(
        first_train_path
    )


    hog_feature_count = len(
        hog_sample
    )


    combined_feature_count = len(
        combined_sample
    )


    print(
        f"RGB features: "
        f"{len(rgb_sample)}"
    )

    print(
        f"LBP features: "
        f"{len(lbp_sample)}"
    )

    print(
        f"HOG features: "
        f"{len(hog_sample)}"
    )

    print(
        f"Combined features: "
        f"{len(combined_sample)}"
    )


    # ========================================================
    # STEP 3 - PROCESS DATASETS
    # ========================================================

    output_files = {

        "train":
            TRAIN_FEATURES_FILE,

        "validation":
            VAL_FEATURES_FILE,

        "test":
            TEST_FEATURES_FILE

    }


    processing_summary = []

    all_failed_images = []


    for split_name in [

        "train",
        "validation",
        "test"

    ]:

        start = time.time()


        feature_df, failed_images = process_split(

            dataframes[
                split_name
            ],

            split_name

        )


        elapsed = (
            time.time()
            - start
        )


        expected_count = (
            EXPECTED_COUNTS[
                split_name
            ]
        )


        if len(feature_df) != expected_count:

            print(
                f"\nERROR: "
                f"{split_name} produced "
                f"{len(feature_df)} feature rows "
                f"instead of {expected_count}."
            )

            sys.exit(1)


        save_features(

            feature_df,

            output_files[
                split_name
            ]

        )


        all_failed_images.extend(

            [

                {
                    "split":
                        split_name,

                    **item

                }

                for item in failed_images

            ]

        )


        processing_summary.append({

            "split":
                split_name,

            "input_images":
                len(
                    dataframes[
                        split_name
                    ]
                ),

            "feature_rows":
                len(feature_df),

            "feature_columns":
                len(feature_df.columns),

            "rgb_features":
                len(rgb_sample),

            "lbp_features":
                len(lbp_sample),

            "hog_features":
                len(hog_sample),

            "combined_features":
                len(combined_sample),

            "failed_images":
                len(failed_images),

            "processing_time_seconds":
                round(
                    elapsed,
                    2
                ),

            "status":
                (
                    "PASS"
                    if not failed_images
                    else "FAIL"
                )

        })


    # ========================================================
    # STEP 4 - SAVE FAILED IMAGE REPORT
    # ========================================================

    if all_failed_images:

        failed_file = (

            OUTPUT_DIR
            / "failed_images.csv"

        )


        pd.DataFrame(
            all_failed_images
        ).to_csv(

            failed_file,

            index=False

        )


        print(
            "\nWARNING:"
        )

        print(
            f"{len(all_failed_images)} "
            "images failed feature extraction."
        )

        print(
            f"Report: {failed_file}"
        )

        sys.exit(1)


    # ========================================================
    # STEP 5 - SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        processing_summary
    )


    summary_df.to_csv(

        SUMMARY_FILE,

        index=False

    )


    print(
        "\nFeature extraction summary saved to:"
    )

    print(
        SUMMARY_FILE
    )


    # ========================================================
    # STEP 6 - SAVE CONFIGURATION
    # ========================================================

    with open(

        CONFIG_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(

            "PHASE 3.3 - "
            "TRADITIONAL ML FEATURE EXTRACTION\n"

        )

        f.write(
            "=" * 65 + "\n\n"
        )


        f.write(
            "Purpose:\n"
        )

        f.write(

            "Convert dermoscopic images into "
            "handcrafted numerical features "
            "for traditional ML models.\n\n"

        )


        f.write(
            "Target models:\n"
        )

        f.write(
            "- Logistic Regression\n"
        )

        f.write(
            "- Support Vector Machine (SVM)\n"
        )

        f.write(
            "- Random Forest\n\n"
        )


        f.write(
            "Input:\n"
        )

        f.write(
            "- RGB images\n"
        )

        f.write(
            "- 224 x 224 pixels\n"
        )

        f.write(
            "- Original images only\n"
        )

        f.write(
            "- No augmentation\n\n"
        )


        f.write(
            "Feature Group 1 - RGB Histogram:\n"
        )

        f.write(
            f"- Bins per channel: {RGB_BINS}\n"
        )

        f.write(
            f"- Red features: {RGB_BINS}\n"
        )

        f.write(
            f"- Green features: {RGB_BINS}\n"
        )

        f.write(
            f"- Blue features: {RGB_BINS}\n"
        )

        f.write(
            f"- Total RGB features: "
            f"{RGB_FEATURE_COUNT}\n\n"
        )


        f.write(
            "Feature Group 2 - LBP:\n"
        )

        f.write(
            f"- Points: {LBP_POINTS}\n"
        )

        f.write(
            f"- Radius: {LBP_RADIUS}\n"
        )

        f.write(
            f"- Method: {LBP_METHOD}\n"
        )

        f.write(
            f"- Total LBP features: "
            f"{LBP_FEATURE_COUNT}\n\n"
        )


        f.write(
            "Feature Group 3 - HOG:\n"
        )

        f.write(
            f"- Orientations: "
            f"{HOG_ORIENTATIONS}\n"
        )

        f.write(
            f"- Pixels per cell: "
            f"{HOG_PIXELS_PER_CELL}\n"
        )

        f.write(
            f"- Cells per block: "
            f"{HOG_CELLS_PER_BLOCK}\n"
        )

        f.write(
            f"- Block normalization: "
            f"{HOG_BLOCK_NORM}\n"
        )

        f.write(
            f"- Total HOG features: "
            f"{hog_feature_count}\n\n"
        )


        f.write(
            "Combined feature vector:\n"
        )

        f.write(
            f"- Total features: "
            f"{combined_feature_count}\n\n"
        )


        f.write(
            "Scaling:\n"
        )

        f.write(
            "- Features are stored UNSCALED.\n"
        )

        f.write(
            "- Scaling will be performed in Phase 3.4.\n"
        )

        f.write(
            "- StandardScaler must be fitted only "
            "on training features.\n"
        )

        f.write(
            "- The same fitted scaler will be applied "
            "to validation/test data.\n\n"
        )


        f.write(
            "Class/label mapping:\n"
        )

        f.write(
            "- Benign = 0\n"
        )

        f.write(
            "- Malignant = 1\n\n"
        )


        f.write(
            "Data leakage protection:\n"
        )

        f.write(
            "- Feature extraction uses each split independently.\n"
        )

        f.write(
            "- No validation/test information is used "
            "to calculate training statistics.\n"
        )

        f.write(
            "- Scaling is intentionally deferred to Phase 3.4.\n"
        )


    print(
        "\nConfiguration saved to:"
    )

    print(
        CONFIG_FILE
    )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 3.3 FEATURE EXTRACTION COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        "\nGenerated files:"
    )

    print(
        f"  [OK] {TRAIN_FEATURES_FILE}"
    )

    print(
        f"  [OK] {VAL_FEATURES_FILE}"
    )

    print(
        f"  [OK] {TEST_FEATURES_FILE}"
    )

    print(
        f"  [OK] {SUMMARY_FILE}"
    )

    print(
        f"  [OK] {CONFIG_FILE}"
    )


    print(
        "\nFeature dimensions:"
    )

    print(
        f"  RGB:      {len(rgb_sample)}"
    )

    print(
        f"  LBP:      {len(lbp_sample)}"
    )

    print(
        f"  HOG:      {len(hog_sample)}"
    )

    print(
        f"  Combined: {len(combined_sample)}"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "The feature CSVs contain UNSCALED features."
    )

    print(
        "Do not manually scale them."
    )

    print(
        "Scaling will be handled in Phase 3.4."
    )


if __name__ == "__main__":

    main()