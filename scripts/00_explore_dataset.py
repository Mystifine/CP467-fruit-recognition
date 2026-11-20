"""
Analyze the Fruits-360 dataset to determine the total number of images
and unique fruit varieties available for each selected class.

Use the scan_all() function to retrieve the dataset information, then group
the images by fruit class to calculate the total image count and number of
distinct varieties. Display a detailed breakdown of how many images belong
to each variety within each class.

This analysis helps identify class imbalances and determine the appropriate
data-splitting strategy. Classes with at least 5 varieties can be split
by variety, while classes with fewer than 5 varieties will use blocks of
consecutive frames to reduce data leakage between development and test sets.

Run this script before selecting images and creating the dataset splits.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import scan_all

df = scan_all()
print("\n=== Totals per class ===")
summary = df.groupby("label").agg(images=("path", "count"), varieties=("variety", "nunique"))
print(summary)

for label, g in df.groupby("label"):
    print(f"\n--- {label}: varieties and image counts ---")
    print(g["variety"].value_counts().to_string())

print("\nWrite down: apples X varieties, bananas Y, oranges Z.")
print("If any class has fewer than 5 varieties, the split script will group by")
print("blocks of consecutive frames instead (mention this in your report).")
