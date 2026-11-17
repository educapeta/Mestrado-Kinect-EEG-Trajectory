import unittest
import torch

from mi_eegnet.data import carrega_config
from mi_eegnet.model import build_model, count_parameters, EEGNetConfigurable


class TestEEGNet(unittest.TestCase):
    def test_parameter_count(self):
        cfg = carrega_config("mi_eegnet/configs/default.yaml")
        model = build_model(cfg)
        n = count_parameters(model)
        self.assertGreaterEqual(n, 5800)
        self.assertLessEqual(n, 6000)
        # valor exato da variante padrao
        self.assertEqual(n, 5870)

    def test_forward_shape(self):
        model = EEGNetConfigurable(n_channels=3, n_times=256, n_classes=2,
                                   classifier_hidden=36)
        x = torch.zeros(4, 3, 256)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
