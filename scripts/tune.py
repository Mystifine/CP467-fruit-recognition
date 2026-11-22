import sys;
import json;
import csv;

import matplotlib.pyplot as plt;
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold;

from src.classifiers import build_X, make_knn;
from src.data import SEED, load_features, FIGURES_DIR, RESULTS_DIR;

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 61, 81]; # k values to try (how many neighbours vote)
SHAPE_WEIGHTS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]; # how much the 3 shape column contribute / weigh

def cv_score(model, X, labels, groups, cv):
  # 5 fold cross validation have 5 sets, pick one for training and the rest for validation
  # this is done on dev
  scores = cross_val_score(model, X, labels, groups=groups, cv=cv, scoring="f1_macro");
  return scores.mean(), scores.std();

def best_row(rows, method):
  # the row with highest mean score for method
  best = None;
  for row in rows:
    if row[0] == method and (best is None or row[3] > best[3]):
      best = row;
  return best;

def main():
  fake = "--fake" in sys.argv; # this is for fake arg
  dev = load_features("dev", fake=fake); # loads feature could be fake

  # group by variety
  varieties = [];
  for g in dev["group"]:
    varieties.append(g.split("#")[0]);

  # updated this to 4 because orange has 4 dev varieties
  cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED); # select 4 splits, 3 go as training 1 is testing
  rows = []; # one row per setting [method, k, shape_weight, mean, std];

  # first we don't use shape because we want to compare shape vs no shape
  X = build_X(dev, False);
  for k in K_VALUES:
    mean, std = cv_score(make_knn(k, False), X, dev["label"], varieties, cv);
    rows.append(["knn_color", k, "", mean, std]); # adds to the row

  # other method is with shape
  X = build_X(dev, True);
  for k in K_VALUES:
    for w in SHAPE_WEIGHTS:
      mean, std = cv_score(make_knn(k, True, w), X, dev["label"], varieties, cv);
      rows.append(["knn_color_shape", k, w, mean, std]);

  best_color = best_row(rows, "knn_color");
  best_shape = best_row(rows, "knn_color_shape");

  # save winners now
  RESULTS_DIR.mkdir(exist_ok=True);
  best = {
    "knn_color": {"k": best_color[1]},
    "knn_color_shape": {"k": best_shape[1], "shape_weight": best_shape[2]},
  };

  with open(RESULTS_DIR / "best_params.json", "w") as f:
    json.dump(best, f, indent=2);

  # save scores
  with open(RESULTS_DIR / "cv_results.cv", "w", newline="") as f:
    writer = csv.writer(f);
    writer.writerow(["method", "k", "shape_weight", "mean_f1", "std_f1"]);
    for row in rows:
      writer.writerow([row[0], row[1], row[2], round(row[3],4), round(row[4],4)]);

  # plot score vs k for each method
  color_scores = [];
  shape_scores = [];
  for row in rows:
    if row[0] == "knn_color":
      color_scores.append(row[3]);
    if row[0] == "knn_color_shape" and row[2] == best_shape[2]:
      shape_scores.append(row[3]); # because there are more for color and shape we only take best
  plt.plot(K_VALUES, color_scores, marker="o", label="KNN color");
  plt.plot(K_VALUES, shape_scores, marker="o", label="KNN color + shape (weight " + str(best_shape[2]) + ")");
  plt.xlabel("K (number of neighbours)");
  plt.ylabel("mean macro-F1 (5-fold CV on dev)");
  plt.legend();
  FIGURES_DIR.mkdir(parents=True, exist_ok=True);
  plt.savefig(FIGURES_DIR / "cv_k_curve.png", dpi=150);

  print("color only:    k =", best_color[1], "| CV macro-F1", round(best_color[3], 3));
  print("color + shape: k =", best_shape[1], "weight =", best_shape[2], "| CV macro-F1", round(best_shape[3], 3));

if __name__ == "__main__":
  main();