import unittest
import numpy as np

from mi_eegnet.train_utils import ChannelScaler


class TestScaler(unittest.TestCase):
    def test_fit_only_train(self):
        rng = np.random.default_rng(0)
        Xtr = rng.normal(10, 2, (10, 3, 256))
        Xte = rng.normal(-100, 5, (5, 3, 256))
        sc = ChannelScaler().fit(Xtr)
        # a media do scaler deve ser a do treino, nao a de Xte
        self.assertTrue(np.allclose(sc.mean, Xtr.mean(axis=(0, 2),
                                                      keepdims=True)))
        Z = sc.transform(Xte)
        self.assertTrue(np.isfinite(Z).all())
        self.assertFalse(np.isclose(Z.mean(), 0.0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
