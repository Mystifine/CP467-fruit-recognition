import csv;
import json;
import sys;
import time;

import matplotlib.pyplot as plt;
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, precision_recall_fscore_support;

from src.classifiers import build_X, hsv_predict, make_knn;
from src.data import CLASSES, FIGURES_DIR, RESULTS_DIR, load_features;

METHODS = ["hsv_rules", "knn_color", "knn_color_shape"];


def main():
  fake = "--fake" in sys.argv;
  dev = load_features("dev", fake=fake);
  test = load_features("test", fake=fake);
  with open(RESULTS_DIR / "best_params.json") as f:
    best = json.load(f);

  y = test["label"];
  n = len(y);
  preds = {};
  predict_ms = {};   # time to classify one image, in milliseconds

  # Method 1
  start = time.perf_counter();
  preds["hsv_rules"] = hsv_predict(test["hue"]);
  predict_ms["hsv_rules"] = (time.perf_counter() - start) / n * 1000;

  # Methods 2 and 3
  for name, use_shape in [("knn_color", False), ("knn_color_shape", True)]:
    k = best[name]["k"];
    weight = best[name].get("shape_weight", 1.0);
    model = make_knn(k, use_shape, weight);
    model.fit(build_X(dev, use_shape), dev["label"]);   
    X_test = build_X(test, use_shape);
    start = time.perf_counter();
    preds[name] = model.predict(X_test);                 # time only the prediction
    predict_ms[name] = (time.perf_counter() - start) / n * 1000;

  # feature extraction time per image (measured by build_features.py), same for all methods
  extract_ms = test["t_extract"].mean() * 1000;

  # metrics table
  RESULTS_DIR.mkdir(exist_ok=True);
  FIGURES_DIR.mkdir(parents=True, exist_ok=True);
  with open(RESULTS_DIR / "metrics.csv", "w", newline="") as f:
    writer = csv.writer(f);
    writer.writerow(["method", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                     "predict_ms", "extract_ms", "total_ms"]);
    for name in METHODS:
      precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds[name], labels=CLASSES, average="macro", zero_division=0);
      writer.writerow([name,
                       round(accuracy_score(y, preds[name]), 4),
                       round(precision, 4), round(recall, 4), round(f1, 4),
                       round(predict_ms[name], 4), round(extract_ms, 4),
                       round(predict_ms[name] + extract_ms, 4)]);
      print("-----", name, "-----");
      print(classification_report(y, preds[name], labels=CLASSES, zero_division=0));

  # one confusion matrix per method
  for name in METHODS:
    fig, ax = plt.subplots(figsize=(4, 4));
    ConfusionMatrixDisplay.from_predictions(y, preds[name], labels=CLASSES, ax=ax, colorbar=False);
    ax.set_title(name);
    fig.savefig(FIGURES_DIR / ("confusion_" + name + ".png"), dpi=150, bbox_inches="tight");
    plt.close(fig);

  # every test prediction, so Person A can find the failures for the report
  with open(RESULTS_DIR / "test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f);
    writer.writerow(["path", "label"] + METHODS);
    for i in range(n):
      writer.writerow([test["path"][i], y[i], preds["hsv_rules"][i],
                       preds["knn_color"][i], preds["knn_color_shape"][i]]);

  print("saved metrics.csv, test_predictions.csv and 3 confusion matrices");


if __name__ == "__main__":
  main();
