import numpy as np;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import StandardScaler, FunctionTransformer;
from sklearn.compose import ColumnTransformer;

def build_X(data, use_shape):
  # it's expected that the data contains hist and shape.
  # hist is a histogram with 64 bins of color data
  # shape is more simple with only 3 features, aspect ratio, circularity and how solid
  if use_shape:
    # hstack combines the vectors into 1 to represent one image
    return np.hstack([data["hist"], data["shape"]]);
  return data["hist"];

N_HIST = 64; # number of histogram bins
def _weight_shape(X, shape_weight):
  # applies the shape weight to all the data points
  X = np.array(X, dtype=np.float64, copy=True);
  X[:, N_HIST:] *= shape_weight; # applies to last 3 for each point
  return X;

def make_knn(k, use_shape, shape_weight=1.0):
  # high level knn overview: we take the extracted features and put them as datapoints in memory
  # we then vote when new datapoint comes in (as a vector) you measure distance to all the points in your data. 
  # example if point has 2 closest to "apple" and 1 "orange" then apple wins and new point is classified as apple
  # the weight matters as there are 64 color items and 3 shape ones
  # updating scale since color histograms are between 0 and 1 we are going to standardize the shape weights

  steps = [];
  if use_shape:
    scale_shape = ColumnTransformer([
      ("hist", "passthrough", list(range(N_HIST))),
      ("shape", StandardScaler(), [N_HIST, N_HIST + 1, N_HIST + 2]),
    ]);
    steps.append(("scale", scale_shape));
    steps.append(("weight", FunctionTransformer(_weight_shape, kw_args={"shape_weight": shape_weight})));
  steps.append(("knn", KNeighborsClassifier(n_neighbors=k)));
  return Pipeline(steps);

