"""
Feature lagging module to prevent look-ahead bias.

CRITICAL: This module ensures that features at time T only use data from time T-1,
preventing the model from having access to future information.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

class FeatureLagger:
    """
    Lag features to prevent look-ahead bias in trading models.

    Look-ahead bias occurs when features calculated at time T include information
    from time T or later. This class ensures all features are properly lagged so
    that predictions at time T only use information available up to time T-1.
    """

    def __init__(self, lag_periods: int = 1):
        """
        Initialize the feature lagger.

        Args:
            lag_periods: Number of periods to lag features (default 1)
        """
        self.lag_periods = lag_periods

    def lag_features(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Lag specified features by lag_periods.

        Args:
            df: DataFrame with features
            feature_columns: List of columns to lag. If None, lags all columns.

        Returns:
            DataFrame with lagged features
        """
        if feature_columns is None:
            feature_columns = df.columns.tolist()

        lagged_df = df.copy()

        for col in feature_columns:
            if col in df.columns:
                lagged_df[col] = df[col].shift(self.lag_periods)
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")

        return lagged_df

    def lag_price_features(self, price_df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Lag all price-based features while preserving metadata columns.

        Args:
            price_df: DataFrame with price features
            exclude_columns: Columns to NOT lag (e.g., 'ticker', 'timestamp')

        Returns:
            DataFrame with lagged price features
        """
        if exclude_columns is None:
            exclude_columns = []

        # Identify feature columns (exclude metadata)
        feature_cols = [col for col in price_df.columns if col not in exclude_columns]

        # Lag only feature columns
        lagged_df = price_df.copy()
        for col in feature_cols:
            lagged_df[col] = price_df[col].shift(self.lag_periods)

        return lagged_df

    def verify_no_leakage(self, original_df: pd.DataFrame, lagged_df: pd.DataFrame,
                         feature_columns: List[str]) -> bool:
        """
        Verify that lagged features don't have look-ahead bias.

        Checks that features at index i in lagged_df equal features at index i-lag_periods
        in original_df.

        Args:
            original_df: Original unlagged DataFrame
            lagged_df: Lagged DataFrame
            feature_columns: Columns to verify

        Returns:
            True if no leakage detected, False otherwise
        """
        for col in feature_columns:
            if col not in original_df.columns or col not in lagged_df.columns:
                continue

            # Check alignment: lagged[i] should equal original[i-lag_periods]
            for i in range(self.lag_periods, len(original_df)):
                original_val = original_df[col].iloc[i - self.lag_periods]
                lagged_val = lagged_df[col].iloc[i]

                # Handle NaN comparison
                if pd.isna(original_val) and pd.isna(lagged_val):
                    continue

                if not np.isclose(original_val, lagged_val, rtol=1e-9, atol=1e-12, equal_nan=True):
                    print(f"Leakage detected in column '{col}' at index {i}!")
                    print(f"  Expected (from index {i-self.lag_periods}): {original_val}")
                    print(f"  Got: {lagged_val}")
                    return False

        print(f"✓ No leakage detected. All {len(feature_columns)} features properly lagged.")
        return True

    def get_aligned_features_and_labels(self, features_df: pd.DataFrame, labels_df: pd.DataFrame,
                                       feature_columns: List[str]) -> tuple:
        """
        Get features and labels with proper temporal alignment.

        Features at time T are lagged to T-1, and paired with labels at time T.
        This ensures predictions use only past information.

        Args:
            features_df: DataFrame with features (indexed by time)
            labels_df: DataFrame with labels (indexed by time)
            feature_columns: Columns to lag

        Returns:
            (lagged_features, aligned_labels) tuple
        """
        # Lag features
        lagged_features = self.lag_features(features_df[feature_columns])

        # Drop rows with NaN from lagging
        lagged_features = lagged_features.dropna()

        # Align labels to match lagged features
        aligned_labels = labels_df.loc[lagged_features.index]

        print(f"Aligned {len(lagged_features)} samples with proper temporal ordering")
        print(f"Dropped {len(features_df) - len(lagged_features)} samples due to lagging")

        return lagged_features, aligned_labels


def demonstrate_lagging():
    """Demonstration of proper feature lagging."""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=10, freq='5min')
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'ema_10': [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]
    }, index=dates)

    print("Original DataFrame:")
    print(df)
    print("\n" + "="*60 + "\n")

    # Apply lagging
    lagger = FeatureLagger(lag_periods=1)
    lagged_df = lagger.lag_features(df, feature_columns=['ema_10'])

    print("Lagged DataFrame (ema_10 shifted by 1):")
    print(lagged_df)
    print("\n" + "="*60 + "\n")

    # Verify
    print("Verification:")
    for i in range(1, len(df)):
        print(f"Index {i}: Original ema_10[{i-1}]={df['ema_10'].iloc[i-1]:.1f} == "
              f"Lagged ema_10[{i}]={lagged_df['ema_10'].iloc[i]:.1f}")

    # Verify no leakage
    print("\n" + "="*60 + "\n")
    lagger.verify_no_leakage(df, lagged_df, ['ema_10'])


if __name__ == "__main__":
    print("Feature Lagger Module - Prevents Look-Ahead Bias")
    print("="*60)
    print("\nDemonstration:\n")
    demonstrate_lagging()
