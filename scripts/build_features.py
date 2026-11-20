
"""
Extract and save colour and shape features from the dataset.

Processes images from the development and test splits, extracts
features for each detected fruit, and saves the results as NPZ files.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.data import resolve
from src.features import extract_features
from src.segment import load_image

# Define the project root, dataset split folder, and feature output folder.
ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "splits"
OUT_DIR = ROOT / "results" / "features"


def read_rows(split):
    """Read the specified dataset split CSV and return its rows as dictionaries."""

    # Locate the CSV file corresponding to the requested split.
    csv_path = SPLIT_DIR / f"{split}.csv"

    # Ensure the dataset has been split before extracting features.
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Must run `python -m src.data` first.")

    # Read image paths, labels, and group information from the CSV.
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_split(split, limit=None):
    """
    Extract features for every valid image in a dataset split.

    Returns extracted feature arrays, skipped image records,
    and the total number of images attempted.
    """

    # Load the image records for the selected split.
    rows = read_rows(split)

    # Optionally restrict processing to a smaller sample for testing.
    if limit:
        rows = rows[:limit]

    # Store extracted features, image metadata, and processing times.
    hist, shape, hue, label, group, path, t_extract = [], [], [], [], [], [], []
    skipped = []

    # Process each image individually.
    for i, row in enumerate(rows, 1):

        # Load the image and record any file reading errors.
        try:
            img = load_image(resolve(row["path"]))
        except Exception as exc:
            skipped.append((row["path"], f"load failed: {exc}"))
            continue

        # Skip images that could not be loaded.
        if img is None:
            skipped.append((row["path"], "load failed"))
            continue

        # Measure the time required to segment the fruit and extract features.
        t0 = time.perf_counter()
        out = extract_features(img)
        dt = time.perf_counter() - t0

        # Skip images where segmentation did not detect a valid fruit.
        if out["contour"] is None:
            skipped.append((row["path"], "no fruit found"))
            continue

        # Store the colour histogram, shape features, and hue fractions.
        hist.append(out["hist"])
        shape.append(out["shape"])
        hue.append(out["hue"])

        # Preserve the original label, group, and path for each image.
        label.append(row["label"])
        group.append(row["group"])
        path.append(row["path"])

        # Record feature extraction time for performance evaluation.
        t_extract.append(dt)

        # Display progress every 500 images or at the final image.
        if i % 500 == 0 or i == len(rows):
            print(f"  {split}: {i}/{len(rows)}")

    # Stop if no usable images were found in the dataset split.
    if not hist:
        raise SystemExit(f"No usable images in split '{split}'.")

    # Convert the collected features and metadata into NumPy arrays.
    # These arrays can be loaded later for model training and evaluation.
    data = {
        "hist": np.array(hist, dtype=np.float64),
        "shape": np.array(shape, dtype=np.float64),
        "hue": np.array(hue, dtype=np.float64),
        "label": np.array(label),
        "group": np.array(group),
        "path": np.array(path),
        "t_extract": np.array(t_extract, dtype=np.float64),
    }

    return data, skipped, len(rows)


def main():
    """Parse command-line arguments and save extracted features for each split."""

    # Configure command-line options for selecting splits and output settings.
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Choose whether to process development, test, or both datasets.
    ap.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both"
    )

    # Allow a limited number of images for quick functionality testing.
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N rows (smoke test)"
    )

    # Allow the output directory to be specified.
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="output folder"
    )

    args = ap.parse_args()

    # Create the output directory if it does not already exist.
    args.out.mkdir(parents=True, exist_ok=True)

    # Determine which dataset splits need to be processed.
    splits = ["dev", "test"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"Extracting features for {split} ...")

        # Extract features and collect information about failed images.
        data, skipped, n_rows = build_split(split, args.limit)

        # Save all feature arrays and metadata in a compressed dataset format.
        out_file = args.out / f"{split}.npz"
        np.savez(out_file, **data)

        # Count the number of successfully processed images for each fruit class.
        n = len(data["label"])
        counts = {
            str(c): int((data["label"] == c).sum())
            for c in sorted(set(data["label"]))
        }

        # Report the output location and processing results.
        print(f"  saved {out_file}")
        print(
            f"  images: {n} of {n_rows}  |  "
            f"skipped: {len(skipped)}  |  per fruit: {counts}"
        )

        # Report the average feature extraction time in milliseconds.
        print(
            f"  mean t_extract: "
            f"{data['t_extract'].mean() * 1000:.2f} ms per image"
        )

        # Display up to 20 skipped images and their failure reasons.
        for p, why in skipped[:20]:
            print(f"    skipped {p}  ({why})")

        # Summarize any additional skipped images.
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")


# Execute feature extraction only when this script is run directly.
if __name__ == "__main__":
    main()
