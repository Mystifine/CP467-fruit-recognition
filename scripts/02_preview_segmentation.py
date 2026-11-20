"""
Visually evaluate the image segmentation process by generating
contour overlays for approximately 20 randomly selected images per fruit class.

Using only images from the development dataset, randomly sample up to 20
images from each class and apply the preprocessing function to generate
segmentation masks and identify fruit contours. Draw the detected contours
in green over the original images to visually assess how accurately the
segmentation process identifies fruit boundaries.

Arrange the processed images into a grid containing 10 columns, adding
blank tiles when necessary to maintain a consistent layout. Save a separate
grid for each fruit class in the results/qualitative/segmentation_check
directory.

This visual inspection helps identify segmentation errors, inaccurate
contour detection, and potential preprocessing issues before proceeding
with feature extraction and model training. The test dataset is excluded
to prevent it from influencing preprocessing decisions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

import config
from src.preprocessing import process

N = 20
COLS = 10
df = pd.read_csv(config.SPLIT_DIR / "dev.csv")      # dev only, never test
out_dir = config.RESULTS_DIR / "qualitative" / "segmentation_check"
out_dir.mkdir(parents=True, exist_ok=True)

for label, g in df.groupby("label"):
    g = g.sample(min(N, len(g)), random_state=config.SEED)
    tiles = []
    for p in g["path"]:
        path = Path(p) if Path(p).is_absolute() else config.ROOT / p
        img, mask, contour = process(path)
        vis = img.copy()
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)
        tiles.append(vis)
    while len(tiles) % COLS:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    grid = np.vstack(rows)
    cv2.imwrite(str(out_dir / f"{label}.png"), grid)
    print("Saved", out_dir / f"{label}.png")
