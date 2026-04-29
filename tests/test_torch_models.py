import unittest


class TestMLPClassifierTorch(unittest.TestCase):
    def test_configured_learning_rate_is_used_by_optimizer(self):
        try:
            from pymegdec.torch_models import MLPClassifierTorch
        except ImportError as exc:
            self.skipTest(f"PyTorch dependencies are not installed: {exc}")

        model = MLPClassifierTorch(
            input_dim=2,
            hidden_dim=3,
            output_dim=2,
            learning_rate=0.123,
        )

        optimizer = model.configure_optimizers()

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.123)


if __name__ == "__main__":
    unittest.main()
