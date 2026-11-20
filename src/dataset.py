"""
Prepare the Fruits-360 dataset for fruit image classification by locating images
from the selected fruit classes, assigning labels, and extracting frame numbers
from filenames. Randomly select approximately N images per class while distributing
samples evenly across available fruit varieties to maintain class balance.

To reduce data leakage, group related images by fruit variety or consecutive
frame blocks, preventing closely related images from being split between
development and testing datasets. Finally, perform a group-aware train/test
split for each class to support more reliable model evaluation.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

IMG_EXT = {".jpg", ".jpeg", ".png"}


def _label_for_folder(name):
    for prefix, label in config.CLASSES.items():
        if name.startswith(prefix + " ") or name == prefix:
            if any(name.startswith(x) for x in config.EXCLUDE_PREFIXES):
                return None
            return label
    return None


def _image_number(filename):
    """'r2_45_100.jpg' / '45_100.jpg' -> 45 (frame number); None if unparseable."""
    m = re.match(r"^(?:r\d*_)?(\d+)_", filename)
    return int(m.group(1)) if m else None


def scan_all(root=None):
    """Return every image of our classes (Training + Test folders pooled)."""
    root = root or config.DATA_ROOT
    rows = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        label = _label_for_folder(folder.name)
        if label is None:
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in IMG_EXT:
                num = _image_number(f.name)
                rows.append({
                    "path": str(f.relative_to(config.ROOT)) if config.ROOT in f.parents else str(f),
                    "label": label,
                    "variety": folder.name,
                    "frame": -1 if num is None else num,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No images found under {root}. Is Fruits-360 unzipped there?")
    # a variety may appear in both Training/ and Test/ folders: keep them as one variety
    return df.drop_duplicates(subset="path").reset_index(drop=True)


def select_images(df, n_per_class=None, seed=None):
    """Pick ~n per class, spread evenly across varieties (round-robin)."""
    n = n_per_class or config.N_PER_CLASS
    rng = np.random.default_rng(seed if seed is not None else config.SEED)
    picked = []
    for label, g in df.groupby("label"):
        queues = {v: list(sub.iloc[rng.permutation(len(sub))].index)
                  for v, sub in g.groupby("variety")}
        chosen = []
        while len(chosen) < n and any(queues.values()):
            for v in list(queues):
                if queues[v] and len(chosen) < n:
                    chosen.append(queues[v].pop(0))
        picked.extend(chosen)
    return df.loc[picked].reset_index(drop=True)


def add_groups(df):
    """Group id used for splitting.

    Classes with enough varieties: group = variety.
    Classes with few varieties:   group = variety + block of consecutive frames,
                                  so near-identical frames stay together.
    """
    groups = []
    for _, r in df.iterrows():
        n_var = df.loc[df.label == r.label, "variety"].nunique()
        if n_var >= config.MIN_VARIETIES_FOR_VARIETY_SPLIT:
            groups.append(r.variety)
        else:
            groups.append(f"{r.variety}#blk{r.frame // config.BLOCK_SIZE}")
    out = df.copy()
    out["group"] = groups
    return out


def grouped_split(df, test_size=None, seed=None):
    """Split each class separately so every class is represented in both sets."""
    test_size = test_size or config.TEST_SIZE
    seed = seed if seed is not None else config.SEED
    dev_parts, test_parts = [], []
    for label, g in df.groupby("label"):
        g = g.reset_index(drop=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        dev_idx, test_idx = next(gss.split(g, groups=g["group"]))
        dev_parts.append(g.iloc[dev_idx])
        test_parts.append(g.iloc[test_idx])
    return (pd.concat(dev_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))