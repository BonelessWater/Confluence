"""
Tests for feature engineering and lagging.

CRITICAL: These tests verify that features don't have look-ahead bias.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.feature_lagger import FeatureLagger


class TestFeatureLagger(unittest.TestCase):
    """Test feature lagging to prevent look-ahead bias."""

    def setUp(self):
        """Create sample data for testing."""
        self.dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        self.df = pd.DataFrame({
            'close': np.arange(100, 200),  # Linear increase
            'ema_10': np.arange(100, 200) * 1.1,  # Slightly higher
            'volume': np.random.randint(1000, 10000, 100)
        }, index=self.dates)

    def test_no_future_leakage(self):
        """CRITICAL: Verify features at time T don't use data from T or later."""
        lagger = FeatureLagger(lag_periods=1)
        lagged_df = lagger.lag_features(self.df, feature_columns=['ema_10'])

        # Verify: lagged feature at time T should equal unlagged feature at T-1
        for i in range(1, len(self.df)):
            original_val = self.df['ema_10'].iloc[i - 1]
            lagged_val = lagged_df['ema_10'].iloc[i]

            self.assertAlmostEqual(
                original_val,
                lagged_val,
                places=6,
                msg=f"Feature leak detected at index {i}! "
                    f"Expected {original_val}, got {lagged_val}"
            )

    def test_first_row_is_nan(self):
        """First row should be NaN after lagging."""
        lagger = FeatureLagger(lag_periods=1)
        lagged_df = lagger.lag_features(self.df, feature_columns=['ema_10'])

        self.assertTrue(
            pd.isna(lagged_df['ema_10'].iloc[0]),
            "First row should be NaN after lagging"
        )

    def test_multiple_periods_lag(self):
        """Test lagging by multiple periods."""
        lag_periods = 5
        lagger = FeatureLagger(lag_periods=lag_periods)
        lagged_df = lagger.lag_features(self.df, feature_columns=['close'])

        # Check lag alignment
        for i in range(lag_periods, len(self.df)):
            original_val = self.df['close'].iloc[i - lag_periods]
            lagged_val = lagged_df['close'].iloc[i]

            self.assertEqual(
                original_val,
                lagged_val,
                msg=f"Lag mismatch at index {i} with lag={lag_periods}"
            )

    def test_verify_no_leakage_function(self):
        """Test the built-in leak verification function."""
        lagger = FeatureLagger(lag_periods=1)
        lagged_df = lagger.lag_features(self.df, feature_columns=['ema_10'])

        # Should pass verification
        result = lagger.verify_no_leakage(
            self.df, lagged_df, ['ema_10']
        )

        self.assertTrue(
            result,
            "Verification should pass for correctly lagged features"
        )

    def test_detect_leakage(self):
        """Test that verification detects incorrectly lagged data."""
        lagger = FeatureLagger(lag_periods=1)

        # Create intentionally wrong lagging (no lag)
        wrong_lagged_df = self.df.copy()

        # Should fail verification
        result = lagger.verify_no_leakage(
            self.df, wrong_lagged_df, ['ema_10']
        )

        self.assertFalse(
            result,
            "Verification should detect non-lagged features"
        )


class TestFeatureAlignment(unittest.TestCase):
    """Test feature and label alignment."""

    def test_aligned_features_and_labels(self):
        """Test that features and labels are properly aligned."""
        # Create data
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        features_df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        }, index=dates)

        labels_df = pd.DataFrame({
            'return': np.random.randn(100)
        }, index=dates)

        # Apply lagging
        lagger = FeatureLagger(lag_periods=1)
        lagged_features, aligned_labels = lagger.get_aligned_features_and_labels(
            features_df, labels_df, ['feature_1', 'feature_2']
        )

        # Check alignment
        self.assertEqual(
            len(lagged_features),
            len(aligned_labels),
            "Features and labels should have same length"
        )

        # Check that indices match
        self.assertTrue(
            (lagged_features.index == aligned_labels.index).all(),
            "Indices should match"
        )

        # Check that first row is dropped (NaN from lagging)
        self.assertEqual(
            lagged_features.index[0],
            dates[1],
            "First date should be dropped due to lagging"
        )


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions."""

    def test_no_future_data_in_ema(self):
        """Verify EMA calculation doesn't use future data."""
        prices = pd.Series([100, 102, 101, 103, 105])

        # Manual EMA calculation with span=3
        ema = prices.ewm(span=3, adjust=False).mean()

        # At index 2, EMA should only use prices[0:3]
        # It should NOT use prices[3] or prices[4]
        self.assertLess(
            ema.iloc[2],
            max(prices.iloc[:3]),
            "EMA should not exceed max of historical prices"
        )

    def test_no_future_data_in_rolling_std(self):
        """Verify rolling std doesn't use future data."""
        prices = pd.Series([100, 105, 95, 110, 90])

        # Rolling std with window=3
        rolling_std = prices.rolling(window=3).std()

        # At index 2, should only use prices[0:3]
        expected_std = prices.iloc[:3].std()

        self.assertAlmostEqual(
            rolling_std.iloc[2],
            expected_std,
            places=6,
            msg="Rolling std should only use historical data"
        )


def run_feature_tests():
    """Run all feature tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*80)
    print("FEATURE ENGINEERING TESTS")
    print("="*80)
    print("\nCRITICAL: These tests verify no look-ahead bias in features\n")

    success = run_feature_tests()

    print("\n" + "="*80)
    if success:
        print("✓ ALL FEATURE TESTS PASSED - No look-ahead bias detected")
    else:
        print("❌ SOME TESTS FAILED - Review feature engineering")
    print("="*80)

    sys.exit(0 if success else 1)
