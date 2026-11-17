import unittest
import numpy as np

from mi_eegnet.train_utils import soft_vote


class TestPrediction(unittest.TestCase):
    def test_soft_vote_returns_one_per_trial(self):
        # 4 trials x 5 janelas x 2 classes
        probs = np.zeros((20, 2))
        probs[:, 0] = 0.8
        probs[:, 1] = 0.2
        groups = np.repeat(np.arange(4), 5)
        uniq, pm, pred = soft_vote(probs, groups)
        self.assertEqual(pm.shape, (4, 2))
        self.assertEqual(pred.shape, (4,))
        self.assertEqual(len(uniq), 4)
        self.assertTrue((pred == 0).all())


if __name__ == "__main__":
    unittest.main()
