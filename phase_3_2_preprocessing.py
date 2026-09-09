"""
Phase 3.2 - Deep Learning Preprocessing & Verification

Project:
Melanoma Skin Lesion Classification (Benign vs Malignant)

Deep Learning Models:
- ResNet50
- EfficientNet-B0
- MobileNetV2
- Vision Transformer (ViT)

Purpose:
    Verify the preprocessing pipeline before model training.

This script:
1. Reads the Phase 3.1 manifests.
2. Verifies image paths.
3. Verifies class and label consistency.
4. Verifies image dimensions and RGB format.
5. Applies training augmentation.
6. Applies ImageNet normalization.
7. Verifies validation/test preprocessing without augmentation.
8. Saves sample augmented images for manual inspection.
9. Saves a preprocessing configuration and verification report.

IMPORTANT:
    - No model is trained in this phase.
    - Original images are NOT modified.
    - Augmentation is only for training data.
    - Saved augmented images are ONLY for visual verification.
"""

from pathlib import Path
import random
import sys

import pandas as pd
from PIL import Image, ImageFile
import torch
from torchvision import transforms


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:/university/summer 26/Data Mining/research project"
)

ARCHIVE_DIR = PROJECT_ROOT / "archive"

PHASE3_DIR = ARCHIVE_DIR / "phase3"

OUTPUT_DIR = PHASE3_DIR / "preprocessing"

AUGMENTED_DIR = OUTPUT_DIR / "augmented_samples"

TRAIN_MANIFEST = PHASE3_DIR / "train_manifest.csv"

VAL_MANIFEST = PHASE3_DIR / "validation_manifest.csv"

TEST_MANIFEST = PHASE3_DIR / "test_manifest.csv"


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUGMENTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# 3. EXPECTED DATASET COUNTS
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


# ============================================================
# 4. EXPECTED LABEL MAPPING
# ============================================================

EXPECTED_LABEL_MAPPING = {

    "Benign": 0,

    "Malignant": 1

}


# ============================================================
# 5. IMAGENET NORMALIZATION
# ============================================================

IMAGENET_MEAN = [

    0.485,
    0.456,
    0.406

]

IMAGENET_STD = [

    0.229,
    0.224,
    0.225

]


# ============================================================
# 6. TRAINING TRANSFORM
# ============================================================

"""
Training augmentation is deliberately conservative.

We avoid aggressive cropping and strong color manipulation
because skin-lesion appearance can contain important visual
information.
"""

train_transform = transforms.Compose([

    # All original images are already 224x224,
    # but this guarantees the required input size.
    transforms.Resize((224, 224)),

    # Geometric augmentation
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        fill=0
    ),

    # Mild color augmentation
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.02
    ),

    # Convert image to tensor.
    # Pixel values become [0, 1].
    transforms.ToTensor(),

    # ImageNet normalization
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])


# ============================================================
# 7. VALIDATION / TEST TRANSFORM
# ============================================================

"""
Validation and test images receive NO augmentation.

Only resizing and normalization are applied.
"""

eval_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])


# ============================================================
# 8. LOAD MANIFEST
# ============================================================

def load_manifest(
    manifest_path,
    split_name
):

    if not manifest_path.exists():

        raise FileNotFoundError(

            f"\n{split_name} manifest not found:\n"
            f"{manifest_path}"

        )

    df = pd.read_csv(manifest_path)

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

        raise ValueError(

            f"\n{split_name} manifest is missing columns: "
            f"{missing_columns}\n"
            f"Available columns: {list(df.columns)}"

        )

    return df


# ============================================================
# 9. VERIFY MANIFEST
# ============================================================

def verify_manifest(
    df,
    split_name
):

    errors = []

    # --------------------------------------------------------
    # Image count
    # --------------------------------------------------------

    actual_count = len(df)

    expected_count = EXPECTED_COUNTS[split_name]

    if actual_count != expected_count:

        errors.append(

            f"{split_name}: expected "
            f"{expected_count} images, "
            f"found {actual_count}"

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


    for class_name, expected in (

        EXPECTED_CLASS_COUNTS[split_name]
        .items()

    ):

        actual = int(

            class_counts.get(
                class_name,
                0
            )

        )

        if actual != expected:

            errors.append(

                f"{split_name}/{class_name}: "
                f"expected {expected}, "
                f"found {actual}"

            )


    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    label_mapping_errors = 0

    for _, row in df.iterrows():

        class_name = str(
            row["class"]
        ).strip()

        try:

            label_value = int(
                row["label"]
            )

        except (ValueError, TypeError):

            label_mapping_errors += 1

            continue


        expected_label = EXPECTED_LABEL_MAPPING.get(
            class_name
        )


        if expected_label is None:

            label_mapping_errors += 1

        elif label_value != expected_label:

            label_mapping_errors += 1


    if label_mapping_errors > 0:

        errors.append(

            f"{split_name}: "
            f"{label_mapping_errors} "
            f"class/label mapping errors"

        )


    # --------------------------------------------------------
    # Image checks
    # --------------------------------------------------------

    missing = 0

    corrupted = 0

    non_rgb = 0

    wrong_dimensions = 0


    for image_path_string in df["filepath"]:

        image_path = Path(
            str(image_path_string)
        )

        # The manifest already contains absolute paths.
        if not image_path.is_absolute():

            image_path = PROJECT_ROOT / image_path


        if not image_path.exists():

            missing += 1

            continue


        try:

            with Image.open(image_path) as img:

                img.load()

                if img.mode != "RGB":

                    non_rgb += 1


                if img.size != (224, 224):

                    wrong_dimensions += 1


        except Exception:

            corrupted += 1


    # --------------------------------------------------------
    # Record errors
    # --------------------------------------------------------

    if missing > 0:

        errors.append(

            f"{split_name}: "
            f"{missing} missing image paths"

        )


    if corrupted > 0:

        errors.append(

            f"{split_name}: "
            f"{corrupted} corrupted/unreadable images"

        )


    if non_rgb > 0:

        errors.append(

            f"{split_name}: "
            f"{non_rgb} non-RGB images"

        )


    if wrong_dimensions > 0:

        errors.append(

            f"{split_name}: "
            f"{wrong_dimensions} images are not 224x224"

        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "split": split_name,

        "image_count": actual_count,

        "benign_count": int(
            class_counts.get(
                "Benign",
                0
            )
        ),

        "malignant_count": int(
            class_counts.get(
                "Malignant",
                0
            )
        ),

        "missing_images": missing,

        "corrupted_images": corrupted,

        "non_rgb_images": non_rgb,

        "wrong_dimensions": wrong_dimensions,

        "label_mapping_errors": label_mapping_errors,

        "status": (
            "PASS"
            if not errors
            else "FAIL"
        ),

        "errors": errors

    }


# ============================================================
# 10. VERIFY TRANSFORM
# ============================================================

def verify_transform(
    image_path,
    transform,
    transform_name
):

    with Image.open(image_path) as img:

        img = img.convert("RGB")

        tensor = transform(img)


    expected_shape = (

        3,
        224,
        224

    )


    if tuple(tensor.shape) != expected_shape:

        raise ValueError(

            f"{transform_name}: "
            f"unexpected tensor shape "
            f"{tuple(tensor.shape)}. "
            f"Expected {expected_shape}."

        )


    if not torch.isfinite(tensor).all():

        raise ValueError(

            f"{transform_name}: "
            "tensor contains NaN or infinite values."

        )


    return tensor


# ============================================================
# 11. REVERSE NORMALIZATION FOR VISUALIZATION
# ============================================================

def tensor_to_display_image(
    tensor
):

    mean = torch.tensor(
        IMAGENET_MEAN
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        IMAGENET_STD
    ).view(
        3,
        1,
        1
    )


    image = (

        tensor.detach().cpu()
        * std
        + mean

    )


    image = torch.clamp(
        image,
        0,
        1
    )


    return transforms.ToPILImage()(
        image
    )


# ============================================================
# 12. SAVE VERIFICATION RESULTS
# ============================================================

def save_verification_results(
    verification_results,
    transform_results
):

    rows = []


    # Manifest results
    for result in verification_results:

        rows.append({

            "check_type": "manifest",

            "split": result["split"],

            "image": "",

            "image_count": result["image_count"],

            "benign_count": result["benign_count"],

            "malignant_count": result["malignant_count"],

            "missing_images": result["missing_images"],

            "corrupted_images": result["corrupted_images"],

            "non_rgb_images": result["non_rgb_images"],

            "wrong_dimensions": result["wrong_dimensions"],

            "label_mapping_errors":
                result["label_mapping_errors"],

            "status": result["status"]

        })


    # Transform results
    for result in transform_results:

        rows.append({

            "check_type": "transform",

            "split": "training_sample",

            "image": result["image"],

            "image_count": "",

            "benign_count": "",

            "malignant_count": "",

            "missing_images": "",

            "corrupted_images": "",

            "non_rgb_images": "",

            "wrong_dimensions": "",

            "label_mapping_errors": "",

            "status": result["status"]

        })


    output_path = (

        OUTPUT_DIR
        / "preprocessing_verification.csv"

    )


    pd.DataFrame(rows).to_csv(

        output_path,
        index=False

    )


    print(
        "\nVerification report saved to:"
    )

    print(output_path)


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 3.2 - "
        "DEEP LEARNING PREPROCESSING VERIFICATION"
    )

    print("=" * 70)


    print("\nModels supported:")

    print("  - ResNet50")

    print("  - EfficientNet-B0")

    print("  - MobileNetV2")

    print("  - Vision Transformer (ViT)")


    print("\nInput size: 224 x 224")

    print("Input format: RGB")

    print("Normalization: ImageNet")

    print(
        "Training augmentation: ENABLED"
    )

    print(
        "Validation augmentation: DISABLED"
    )

    print(
        "Test augmentation: DISABLED"
    )


    # ========================================================
    # STEP 1
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "STEP 1: VERIFYING PHASE 3.1 MANIFESTS"
    )

    print("-" * 70)


    manifests = {

        "train": TRAIN_MANIFEST,

        "validation": VAL_MANIFEST,

        "test": TEST_MANIFEST

    }


    loaded = {}

    verification_results = []


    for split_name, manifest_path in manifests.items():

        df = load_manifest(
            manifest_path,
            split_name
        )


        loaded[split_name] = df


        result = verify_manifest(

            df,
            split_name

        )


        verification_results.append(
            result
        )


        print(
            f"\n{split_name.upper()}: "
            f"{result['image_count']} | "
            f"Benign={result['benign_count']} | "
            f"Malignant={result['malignant_count']} | "
            f"Missing={result['missing_images']} | "
            f"Corrupt={result['corrupted_images']} | "
            f"Label errors={result['label_mapping_errors']} | "
            f"Status={result['status']}"
        )


        for error in result["errors"]:

            print(
                f"  ERROR: {error}"
            )


    # ========================================================
    # STOP IF VERIFICATION FAILED
    # ========================================================

    failed = [

        result
        for result in verification_results
        if result["status"] == "FAIL"

    ]


    if failed:

        save_verification_results(
            verification_results,
            []
        )


        print(
            "\n" + "=" * 70
        )

        print(
            "VERIFICATION FAILED"
        )

        print(
            "Fix the issues above before continuing."
        )

        print(
            "=" * 70
        )

        sys.exit(1)


    print(
        "\nAll manifest and image checks passed."
    )


    # ========================================================
    # STEP 2
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "STEP 2: VERIFYING PYTORCH TRANSFORMS"
    )

    print("-" * 70)


    train_df = loaded["train"]


    # Select 3 Benign + 3 Malignant
    # for preprocessing verification.

    selected_images = []


    for class_name in [

        "Benign",
        "Malignant"

    ]:

        class_df = train_df[
            train_df["class"].astype(str).str.strip()
            == class_name
        ]


        sample = class_df.sample(

            n=min(
                3,
                len(class_df)
            ),

            random_state=SEED

        )


        for _, row in sample.iterrows():

            image_path = Path(
                str(row["filepath"])
            )


            selected_images.append(

                (
                    class_name,
                    image_path
                )

            )


    transform_results = []


    for class_name, image_path in selected_images:

        if not image_path.exists():

            print(
                f"  SKIP: {image_path}"
            )

            continue


        # Validation/test-style transform
        eval_tensor = verify_transform(

            image_path,
            eval_transform,
            "evaluation_transform"

        )


        # Training transform
        train_tensor = verify_transform(

            image_path,
            train_transform,
            "training_transform"

        )


        transform_results.append({

            "class": class_name,

            "image": str(image_path),

            "eval_shape":
                str(tuple(eval_tensor.shape)),

            "train_shape":
                str(tuple(train_tensor.shape)),

            "eval_min":
                float(eval_tensor.min()),

            "eval_max":
                float(eval_tensor.max()),

            "train_min":
                float(train_tensor.min()),

            "train_max":
                float(train_tensor.max()),

            "status": "PASS"

        })


        print(
            f"\n  {class_name}: "
            f"{image_path.name}"
        )

        print(
            f"    Evaluation tensor: "
            f"{tuple(eval_tensor.shape)}"
        )

        print(
            f"    Training tensor: "
            f"{tuple(train_tensor.shape)}"
        )

        print(
            "    Status: PASS"
        )


    # ========================================================
    # STEP 3
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "STEP 3: SAVING AUGMENTATION SAMPLES"
    )

    print("-" * 70)


    visualization_count = 3


    for class_name, image_path in selected_images:

        if not image_path.exists():

            continue


        with Image.open(image_path) as img:

            original = img.convert("RGB")


        original_name = image_path.stem


        # Save original image
        original.save(

            AUGMENTED_DIR
            / f"{class_name}_"
              f"{original_name}_original.jpg",

            quality=95

        )


        # Generate 3 augmented versions
        for i in range(
            visualization_count
        ):

            random.seed(
                SEED + i
            )


            augmented_tensor = train_transform(
                original
            )


            augmented_image = (
                tensor_to_display_image(
                    augmented_tensor
                )
            )


            output_path = (

                AUGMENTED_DIR
                / f"{class_name}_"
                  f"{original_name}_"
                  f"augmented_{i+1}.jpg"

            )


            augmented_image.save(

                output_path,

                quality=95

            )


    print(
        "\nSample images saved to:"
    )

    print(AUGMENTED_DIR)


    # ========================================================
    # STEP 4
    # ========================================================

    save_verification_results(

        verification_results,

        transform_results

    )


    # ========================================================
    # STEP 5 - SAVE CONFIGURATION
    # ========================================================

    config_path = (

        OUTPUT_DIR
        / "preprocessing_config.txt"

    )


    with open(

        config_path,
        "w",
        encoding="utf-8"

    ) as f:

        f.write(
            "PHASE 3.2 - "
            "DEEP LEARNING PREPROCESSING CONFIGURATION\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )


        f.write(
            "Models:\n"
        )

        f.write(
            "- ResNet50\n"
        )

        f.write(
            "- EfficientNet-B0\n"
        )

        f.write(
            "- MobileNetV2\n"
        )

        f.write(
            "- Vision Transformer (ViT)\n\n"
        )


        f.write(
            "Input:\n"
        )

        f.write(
            "- RGB\n"
        )

        f.write(
            "- 224 x 224 pixels\n\n"
        )


        f.write(
            "Training augmentation:\n"
        )

        f.write(
            "- RandomHorizontalFlip(p=0.5)\n"
        )

        f.write(
            "- RandomVerticalFlip(p=0.5)\n"
        )

        f.write(
            "- RandomRotation(degrees=15)\n"
        )

        f.write(
            "- RandomAffine(translation=5%, "
            "scale=0.95-1.05)\n"
        )

        f.write(
            "- ColorJitter("
            "brightness=0.10, "
            "contrast=0.10, "
            "saturation=0.10, "
            "hue=0.02)\n\n"
        )


        f.write(
            "Validation/Test augmentation:\n"
        )

        f.write(
            "- None\n\n"
        )


        f.write(
            "Normalization:\n"
        )

        f.write(
            f"- Mean: {IMAGENET_MEAN}\n"
        )

        f.write(
            f"- Std: {IMAGENET_STD}\n\n"
        )


        f.write(
            "Class/Label mapping:\n"
        )

        f.write(
            "- Benign = 0\n"
        )

        f.write(
            "- Malignant = 1\n\n"
        )


        f.write(
            "Reproducibility seed:\n"
        )

        f.write(
            f"- {SEED}\n\n"
        )


        f.write(
            "Important:\n"
        )

        f.write(
            "- Original images are not modified.\n"
        )

        f.write(
            "- Augmentation should be applied "
            "on-the-fly during model training.\n"
        )

        f.write(
            "- Validation and test images must "
            "remain unaugmented.\n"
        )

        f.write(
            "- Saved augmented images are for "
            "visual verification only.\n"
        )


    print(
        "\nConfiguration saved to:"
    )

    print(config_path)


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 3.2 VERIFICATION PASSED"
    )

    print(
        "=" * 70
    )


    print(
        "\nVerified:"
    )

    print(
        "  [OK] Train manifest"
    )

    print(
        "  [OK] Validation manifest"
    )

    print(
        "  [OK] Test manifest"
    )

    print(
        "  [OK] Image paths"
    )

    print(
        "  [OK] Image readability"
    )

    print(
        "  [OK] RGB format"
    )

    print(
        "  [OK] 224x224 dimensions"
    )

    print(
        "  [OK] Benign/Malignant counts"
    )

    print(
        "  [OK] Class/label mapping"
    )

    print(
        "  [OK] Training augmentation"
    )

    print(
        "  [OK] Validation/test preprocessing"
    )

    print(
        "  [OK] ImageNet normalization"
    )

    print(
        "  [OK] Tensor shape: 3x224x224"
    )

    print(
        "  [OK] Original images untouched"
    )


    print(
        "\nNext step:"
    )

    print(
        "Manually inspect the images in:"
    )

    print(AUGMENTED_DIR)


    print(
        "\nDo NOT begin model training until "
        "the augmentation samples have been inspected."
    )


# ============================================================
# 14. RUN
# ============================================================

if __name__ == "__main__":

    main()