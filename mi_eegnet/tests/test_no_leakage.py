import unittest
import numpy as np

from mi_eegnet.train_utils import group_folds, split_trials


class TestNoLeakage(unittest.TestCase):
    def test_split_trials_disjoint(self):
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        tr, te = split_trials(y, n_test=4, seed=42)
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertEqual(len(tr) + len(te), len(y))

    def test_group_folds_disjoint(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(20), 5)
        y = np.repeat(rng.integers(0, 2, 20), 5)
        folds = group_folds(y, groups, n_splits=5, seed=42)
        for tr, te in folds:
            self.assertEqual(len(set(groups[tr]) & set(groups[te])), 0)


if __name__ == "__main__":
    unittest.main()
