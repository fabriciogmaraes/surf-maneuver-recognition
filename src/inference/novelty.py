"""Flag clips that look unlike every training clip, by k-nearest-neighbor distance in feature space."""
import numpy as np


def _unit(X):
    """Rows scaled to unit length, so a dot product is a cosine similarity."""
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


class KnnNovelty:
    """Mean cosine distance to the k nearest training clips, with a threshold taken from the training set.

    The reference distance of each training clip is measured against clips from OTHER sessions only:
    same-session neighbors are near-duplicates and would make every clip look familiar.
    """

    def __init__(self, k=10, quantile=0.95):
        self.k = k
        self.quantile = quantile

    def fit(self, X, groups):
        Z = _unit(X)
        groups = np.asarray(groups)
        distances = 1 - Z @ Z.T
        distances[groups[:, None] == groups[None, :]] = np.inf
        self.train_distance_ = np.sort(distances, axis=1)[:, :self.k].mean(axis=1)
        self.threshold_ = float(np.quantile(self.train_distance_, self.quantile))
        self.reference_ = Z.astype(np.float16)   # half precision keeps the saved file small
        return self

    def distance(self, X):
        """Mean cosine distance from each row of X to its k nearest training clips."""
        similarity = _unit(X) @ self.reference_.astype(np.float32).T
        return np.sort(1 - similarity, axis=1)[:, :self.k].mean(axis=1)


def add_knn_distance(windows, novelty):
    """Add a `knn_distance` column to the windows of `label_windows` (NaN where no surfer was found)."""
    has_surfer = windows["features"].notna().to_numpy() if "features" in windows else np.zeros(len(windows), bool)
    distance = np.full(len(windows), np.nan)
    if has_surfer.any():
        distance[has_surfer] = novelty.distance(np.stack(windows.loc[has_surfer, "features"]))
    windows["knn_distance"] = distance
    return windows
