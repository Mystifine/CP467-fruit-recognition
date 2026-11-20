"""
Select approximately 300 images per fruit class and divide the
selected images into development and testing datasets.

First, scan the Fruits-360 dataset and randomly select images from each
fruit class while maintaining a balanced distribution across available
varieties. Assign group identifiers based on fruit varieties or consecutive
frame blocks to prevent closely related images from being separated.

Next, perform a group-aware split to create development and test datasets,
ensuring that images belonging to the same group remain in one dataset.
Verify that no groups overlap between the two sets to reduce data leakage
and support reliable model evaluation.

Finally, save the resulting datasets as dev.csv and test.csv in the
data/splits directory. Display the total number of selected images, the
class distribution in each dataset, and the number of groups assigned
to each split for verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import scan_all, select_images, add_groups, grouped_split

df = scan_all()
df = select_images(df)
df = add_groups(df)
dev, test = grouped_split(df)

# Safety check: no group may appear in both sets
overlap = set(dev["group"]) & set(test["group"])
assert not overlap, f"Leakage! Groups in both sets: {overlap}"

config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
dev.to_csv(config.SPLIT_DIR / "dev.csv", index=False)
test.to_csv(config.SPLIT_DIR / "test.csv", index=False)

print("Selected:", len(df), "images")
print("\nDev counts:\n", dev["label"].value_counts().to_string())
print("\nTest counts:\n", test["label"].value_counts().to_string())
print("\nGroups per class (dev / test):")
for label in sorted(df.label.unique()):
    print(f"  {label}: {dev[dev.label==label].group.nunique()} / {test[test.label==label].group.nunique()}")
print("\nSaved to", config.SPLIT_DIR)
