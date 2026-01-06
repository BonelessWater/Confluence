"""
Tests for trading models.

These tests verify that models implement the correct interface
and produce reasonable outputs.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.linear_model import LinearModel
from src.models.xgboost_model import XGBoostModel


class TestModelInterface(unittest.TestCase):
    """Test that models implement the required interface."""

    def setUp(self):
        """Create synthetic data for testing."""
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 50

        # Create features and target
        self.X = np.random.randn(self.n_samples, self.n_features)
        # Simple linear relationship for testing
        true_coeffs = np.random.randn(self.n_features) * 0.1
        self.y = self.X @ true_coeffs + np.random.randn(self.n_samples) * 0.01

        # Split data
        split_idx = int(self.n_samples * 0.7)
        self.X_train, self.X_val = self.X[:split_idx], self.X[split_idx:]
        self.y_train, self.y_val = self.y[:split_idx], self.y[split_idx:]

    def test_linear_model_interface(self):
        """Test LinearModel implements required interface."""
        model = LinearModel(name="Test_Linear", n_features=20, alpha=1.0)

        # Test fit
        results = model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertTrue(model.is_fitted)
        self.assertIn('train_metrics', results)
        self.assertIn('val_metrics', results)

        # Test predict
        predictions = model.predict(self.X_val)
        self.assertEqual(len(predictions), len(self.X_val))
        self.assertFalse(np.any(np.isnan(predictions)))

        # Test evaluate
        metrics = model.evaluate(self.X_val, self.y_val)
        self.assertIn('correlation', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('directional_accuracy', metrics)

    def test_xgboost_model_interface(self):
        """Test XGBoostModel implements required interface."""
        model = XGBoostModel(
            name="Test_XGBoost",
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1
        )

        # Test fit
        results = model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertTrue(model.is_fitted)
        self.assertIn('train_metrics', results)
        self.assertIn('val_metrics', results)

        # Test predict
        predictions = model.predict(self.X_val)
        self.assertEqual(len(predictions), len(self.X_val))
        self.assertFalse(np.any(np.isnan(predictions)))

        # Test evaluate
        metrics = model.evaluate(self.X_val, self.y_val)
        self.assertIn('correlation', metrics)


class TestModelPerformance(unittest.TestCase):
    """Test that models produce reasonable predictions."""

    def setUp(self):
        """Create data with known relationship."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 100

        # Create features
        X = np.random.randn(n_samples, n_features)

        # Create target with strong linear relationship to first 5 features
        true_coeffs = np.zeros(n_features)
        true_coeffs[:5] = [0.5, 0.3, 0.2, 0.1, 0.05]

        y = X @ true_coeffs + np.random.randn(n_samples) * 0.01

        # Split
        split_idx = int(n_samples * 0.7)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]

    def test_linear_model_learns_relationship(self):
        """Test that LinearModel can learn the relationship."""
        model = LinearModel(n_features=50, alpha=1.0)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        metrics = model.evaluate(self.X_val, self.y_val)

        # Should have good correlation since relationship is linear
        self.assertGreater(
            metrics['correlation'],
            0.8,
            f"Linear model should learn linear relationship "
            f"(got correlation {metrics['correlation']:.4f})"
        )

    def test_xgboost_model_learns_relationship(self):
        """Test that XGBoostModel can learn the relationship."""
        model = XGBoostModel(max_depth=5, n_estimators=100, learning_rate=0.1)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        metrics = model.evaluate(self.X_val, self.y_val)

        # Should have good correlation
        self.assertGreater(
            metrics['correlation'],
            0.8,
            f"XGBoost should learn relationship "
            f"(got correlation {metrics['correlation']:.4f})"
        )

    def test_models_dont_overfit_severely(self):
        """Test that models don't overfit too much."""
        # Linear model
        linear = LinearModel(n_features=50, alpha=1.0)
        linear.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        train_corr_linear = linear.evaluate(self.X_train, self.y_train)['correlation']
        val_corr_linear = linear.evaluate(self.X_val, self.y_val)['correlation']

        # Gap should be reasonable (< 0.2)
        gap_linear = train_corr_linear - val_corr_linear
        self.assertLess(
            gap_linear,
            0.2,
            f"Linear model overfitting: train={train_corr_linear:.4f}, "
            f"val={val_corr_linear:.4f}, gap={gap_linear:.4f}"
        )

        # XGBoost model
        xgb_model = XGBoostModel(max_depth=3, n_estimators=50)  # Conservative params
        xgb_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        train_corr_xgb = xgb_model.evaluate(self.X_train, self.y_train)['correlation']
        val_corr_xgb = xgb_model.evaluate(self.X_val, self.y_val)['correlation']

        gap_xgb = train_corr_xgb - val_corr_xgb
        self.assertLess(
            gap_xgb,
            0.3,
            f"XGBoost overfitting: train={train_corr_xgb:.4f}, "
            f"val={val_corr_xgb:.4f}, gap={gap_xgb:.4f}"
        )


class TestModelDeterminism(unittest.TestCase):
    """Test that models produce consistent results."""

    def setUp(self):
        """Create small dataset."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 20)
        self.y_train = np.random.randn(100)
        self.X_val = np.random.randn(30, 20)
        self.y_val = np.random.randn(30)

    def test_linear_model_deterministic(self):
        """Test that LinearModel produces same results with same data."""
        model1 = LinearModel(n_features=10, alpha=1.0)
        model1.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        pred1 = model1.predict(self.X_val)

        model2 = LinearModel(n_features=10, alpha=1.0)
        model2.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        pred2 = model2.predict(self.X_val)

        np.testing.assert_array_almost_equal(
            pred1, pred2,
            err_msg="LinearModel should be deterministic"
        )


def run_model_tests():
    """Run all model tests."""
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*80)
    print("MODEL INTERFACE AND PERFORMANCE TESTS")
    print("="*80)
    print("\nTesting model implementations\n")

    success = run_model_tests()

    print("\n" + "="*80)
    if success:
        print("✓ ALL MODEL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)

    sys.exit(0 if success else 1)
