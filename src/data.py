import csv;
import random;
import sys;
from pathlib import Path;
import numpy as np;

CLASSES = ["apple", "banana", "orange"];
DATA_DIR = Path("data/fruits-360_100x100/fruits-360");   # folder that holds Training/ and Test/
SPLITS_DIR = Path("splits");
FEATURES_DIR = Path("results/features");
RESULTS_DIR = Path("results");
FIGURES_DIR = Path("results/figures");
SEED = 467;            # same seed allows for same random split on every computer
EXCLUDE = ["Orange peeled 1"];          # ignoring these because they aren't really good images

def resolve(rel_path):
  # paths in csvs look like "Training/Apple Red 1/r_123_100.jpg" 
  return DATA_DIR / rel_path;

def get_fruit(folder_name):
  # use folder prefix to get fruit name, example "Apple Red 1" gets "apple"
  first_word = folder_name.split()[0].lower();
  if first_word in CLASSES:
    return first_word;
  return None;

def list_images():
  # returns a dict of {fruit: {variety: [relative paths]}} with Training and Test merged
  images = {};
  for part in ["Training", "Test"]:
    for folder in (DATA_DIR / part).iterdir():
      if not folder.is_dir():
        continue; # skip stray files, only folders hold images
      fruit = get_fruit(folder.name);
      if fruit is None:
        continue; # skip the ones we dont want
      if folder.name in EXCLUDE:
        continue; # skip specific folders

      if fruit not in images:
        images[fruit] = {};
      if folder.name not in images[fruit]:
        images[fruit][folder.name] = [];

      for img in folder.iterdir():
        if img.suffix.lower() != ".jpg":
          continue; # we only want .jpg images just a guard
        path = part + "/" + folder.name + "/" + img.name;
        images[fruit][folder.name].append(path);
  return images;

def print_varieties():
  # shows every variety found and how many images it has, to decide what goes in EXCLUDE
  images = list_images();
  for fruit in CLASSES:
    for variety in sorted(images[fruit].keys()):
      print(fruit, "|", variety, "|", len(images[fruit][variety]), "images");

def frame_number(path):
  # "Training/Apple Red 1/r_123_100.jpg" -> 123
  # the reason we want framae number is because the images are from a video of a rotating fruit
  # frames that are close look identical so we want to gap them or chunk em up
  name = Path(path).stem;
  parts = name.split("_");
  return int(parts[-2]); # second last is frame nmber

def video_order(path):
  # video order matters
  # 57_100 would be first vid
  # r_57_100 is another one but flipped
  # r2_57_100 is anotehr one but flipped again
  name = Path(path).stem;
  parts = name.split("_");
  if len(parts) == 3:
    video = parts[0];
  else:
    video = "";
  part = path.split("/")[0];
  return (part, video, frame_number(path)); # tuple for comparison

def pick_some(paths, how_many, rng):
  # select some since each folder has a lot of frames
  if len(paths) <= how_many:
    return paths;
  positions = rng.sample(range(len(paths)), how_many);
  positions.sort();
  chosen = [];
  for i in positions:
    chosen.append(paths[i]);
  return chosen;

def add_rows(rows, paths, fruit, variety, is_test):
  # one row per image: path, label, variety, group
  # dev images: every 20 frames in a row share a group (used by cross-validation)
  for i in range(len(paths)):
    if is_test:
      group = variety + "#test";
    else:
      group = variety + "#" + str(i // 20);
    rows.append([paths[i], fruit, variety, group]);


def make_split(per_fruit=300, test_fraction=0.2, gap=10):
  # 80% to build 20% to test
  rng = random.Random(SEED);
  dev = [];
  test = [];
  images = list_images();

  for fruit in CLASSES:
    varieties = sorted(images[fruit].keys());
    rng.shuffle(varieties);
    per_variety = per_fruit // len(varieties); # 100 per

    if len(varieties) >= 2:
      # there are many different type of varieties of apples/fruits
      # rule 1: several varieties -> some whole varieties go to test, multiply by test_fraction
      how_many_test = max(1, round(len(varieties) * test_fraction));
      test_varieties = varieties[:how_many_test];

      for v in varieties:
        paths = sorted(images[fruit][v], key=video_order);
        paths = pick_some(paths, per_variety, rng);
        if v in test_varieties:
          add_rows(test, paths, fruit, v, True);
        else:
          add_rows(dev, paths, fruit, v, False);
      
      print(fruit, "-> test varieties:", test_varieties);

    else:
      # otherwise if there is only one variety -> first 80% of frames = dev, last 20% = test,
      # and drop `gap` frames in between so neighbouring frames can't leak
      v = varieties[0];
      paths = sorted(images[fruit][v], key=video_order);
      paths = pick_some(paths, per_fruit, rng);
      cut = int(len(paths) * (1 - test_fraction));

      add_rows(dev, paths[:cut - gap], fruit, v, False);
      add_rows(test, paths[cut:], fruit, v, True);
      print(fruit, "-> only one variety, split by frame order");

  # no image may be in both splits
  dev_paths = set();
  for row in dev:
    dev_paths.add(row[0]);
  for row in test:
    assert row[0] not in dev_paths, "leak: " + row[0] + " is in both splits";

  save_csv("dev", dev);
  save_csv("test", test);
  print("dev:", len(dev), "images | test:", len(test), "images");


def save_csv(name, rows):
  SPLITS_DIR.mkdir(exist_ok=True);
  with open(SPLITS_DIR / (name + ".csv"), "w", newline="") as f:
    writer = csv.writer(f);
    writer.writerow(["path", "label", "variety", "group"]);
    writer.writerows(rows);


def read_split(name):
  # read splits/dev.csv or splits/test.csv back as a list of dicts
  with open(SPLITS_DIR / (name + ".csv")) as f:
    return list(csv.DictReader(f));


def load_features(split, fake=False):
  # returns a dict of arrays: hist, shape, hue, label, group, path, t_extract
  if fake:
    return fake_features(split);
  data = np.load(FEATURES_DIR / (split + ".npz"));
  result = {};
  for key in data.files:
    result[key] = data[key];
  return result;


def fake_features(split, n=200):
  # made-up numbers in the real format, for testing before the real features exist
  if split == "dev":
    rng = np.random.default_rng(SEED);
  else:
    rng = np.random.default_rng(SEED + 1);

  spike = {"apple": 3, "banana": 11, "orange": 7};    # histogram bin where each fruit's colour is
  shape = {"apple": [1.05, 0.87, 0.98], "banana": [2.4, 0.3, 0.55], "orange": [1.0, 0.9, 0.99]};
  hue = {"apple": [0.7, 0.1, 0.1, 0.1], "banana": [0.0, 0.1, 0.8, 0.1], "orange": [0.05, 0.9, 0.05, 0.0]};

  hists, shapes, hues, labels, groups, paths, times = [], [], [], [], [], [], [];
  for fruit in CLASSES:
    for i in range(n):
      h = rng.random(64) * 0.05;                     # a little of every colour (noise)
      h[spike[fruit]] = h[spike[fruit]] + 1;         # plus a big spike for this fruit
      hists.append(h / h.sum());
      shapes.append(np.array(shape[fruit]) + rng.normal(0, 0.06, 3));
      hues.append(np.array(hue[fruit]) + rng.normal(0, 0.05, 4));
      labels.append(fruit);
      groups.append("fake " + fruit + "#" + str(i // 20));
      paths.append("fake/" + fruit + "/" + str(i) + ".jpg");
      times.append(0.002);

  return {
    "hist": np.array(hists),
    "shape": np.array(shapes),
    "hue": np.array(hues),
    "label": np.array(labels),
    "group": np.array(groups),
    "path": np.array(paths),
    "t_extract": np.array(times),
  };


if __name__ == "__main__":
  # python -m src.data list: show varieties and image counts
  # python -m src.data: make splits/dev.csv and splits/test.csv
  if len(sys.argv) > 1 and sys.argv[1] == "list":
    print_varieties();
  else:
    make_split();