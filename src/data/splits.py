"""Group-aware cross-validation folds with real randomness across seeds."""
import numpy as np


def random_group_folds(y, groups, n_folds=5, seed=0):
    """Yield (train_idx, test_idx) with every group kept on one side.

    Groups are visited in a random order and each one goes to the fold where it brings the
    class counts closest to an even share. Unlike sklearn's StratifiedGroupKFold, whose group
    order is mostly fixed by a sort, every seed gives a different partition.
    """
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    classes = np.unique(y)
    target = np.array([(y == c).sum() for c in classes]) / n_folds   # ideal clips per class per fold

    fold_counts = np.zeros((n_folds, len(classes)))
    fold_of_group = {}
    for g in rng.permutation(np.unique(groups)):
        counts = np.array([((groups == g) & (y == c)).sum() for c in classes])
        # Imbalance of each fold before and after receiving this group (relative squared distance to target)
        before = (((fold_counts - target) / target) ** 2).sum(axis=1)
        after = (((fold_counts + counts - target) / target) ** 2).sum(axis=1)
        cost = after - before
        best = rng.choice(np.flatnonzero(np.isclose(cost, cost.min())))   # random tie-break
        fold_counts[best] += counts
        fold_of_group[g] = best

    fold = np.array([fold_of_group[g] for g in groups])
    for f in range(n_folds):
        yield np.flatnonzero(fold != f), np.flatnonzero(fold == f)