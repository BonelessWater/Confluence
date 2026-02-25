"""
Feature engineering module for tweet-based trading strategy.
Calculates technical indicators, volatility metrics, and embeddings features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering for tweet and price data.
    """

    def __init__(self):
        self.feature_columns = []

    def calculate_ema(self, series: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages for multiple periods."""
        emas = {}
        for period in periods:
            emas[f'ema_{period}'] = series.ewm(span=period, adjust=False).mean()
        return pd.DataFrame(emas)

    def calculate_volatility(self, series: pd.Series, windows: List[int]) -> pd.DataFrame:
        """Calculate rolling volatility for multiple windows."""
        vols = {}
        for window in windows:
            vols[f'vol_{window}'] = series.pct_change().rolling(window=window).std()
        return pd.DataFrame(vols)

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist
        })

    def calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()

        return pd.DataFrame({
            'bb_upper': sma + (std * num_std),
            'bb_middle': sma,
            'bb_lower': sma - (std * num_std),
            'bb_width': (std * num_std * 2) / sma
        })

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_momentum(self, series: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Calculate momentum for multiple periods."""
        moms = {}
        for period in periods:
            moms[f'momentum_{period}'] = series.pct_change(periods=period)
        return pd.DataFrame(moms)

    def calculate_price_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive price-based features.

        Args:
            price_df: DataFrame with OHLC price data

        Returns:
            DataFrame with all calculated features
        """
        features = pd.DataFrame(index=price_df.index)

        close = price_df['close']
        high = price_df['high']
        low = price_df['low']

        print("Calculating EMAs...")
        # Multiple timeframe EMAs
        ema_periods = [5, 10, 20, 50, 100, 200]
        ema_features = self.calculate_ema(close, ema_periods)
        features = pd.concat([features, ema_features], axis=1)

        # EMA crossovers
        features['ema_cross_5_20'] = ema_features['ema_5'] - ema_features['ema_20']
        features['ema_cross_10_50'] = ema_features['ema_10'] - ema_features['ema_50']
        features['ema_cross_50_200'] = ema_features['ema_50'] - ema_features['ema_200']

        print("Calculating volatility...")
        # Volatility for different windows
        vol_windows = [5, 10, 20, 50, 100]
        vol_features = self.calculate_volatility(close, vol_windows)
        features = pd.concat([features, vol_features], axis=1)

        # Volatility ratios
        features['vol_ratio_5_20'] = vol_features['vol_5'] / vol_features['vol_20']
        features['vol_ratio_20_50'] = vol_features['vol_20'] / vol_features['vol_50']

        print("Calculating technical indicators...")
        # RSI for multiple periods
        features['rsi_14'] = self.calculate_rsi(close, 14)
        features['rsi_7'] = self.calculate_rsi(close, 7)
        features['rsi_21'] = self.calculate_rsi(close, 21)

        # MACD
        macd_features = self.calculate_macd(close)
        features = pd.concat([features, macd_features], axis=1)

        # Bollinger Bands
        bb_features = self.calculate_bollinger_bands(close)
        features = pd.concat([features, bb_features], axis=1)

        # Price position within Bollinger Bands
        features['bb_position'] = (close - bb_features['bb_lower']) / (bb_features['bb_upper'] - bb_features['bb_lower'])

        # ATR
        features['atr_14'] = self.calculate_atr(high, low, close, 14)
        features['atr_7'] = self.calculate_atr(high, low, close, 7)

        # Momentum
        momentum_periods = [1, 3, 5, 10, 20]
        momentum_features = self.calculate_momentum(close, momentum_periods)
        features = pd.concat([features, momentum_features], axis=1)

        # Price ratios
        features['high_low_ratio'] = high / low
        features['close_open_ratio'] = close / price_df['open']

        # Distance from highs/lows
        features['dist_from_20h'] = (close - high.rolling(20).max()) / close
        features['dist_from_20l'] = (close - low.rolling(20).min()) / close
        features['dist_from_50h'] = (close - high.rolling(50).max()) / close
        features['dist_from_50l'] = (close - low.rolling(50).min()) / close

        # Volume features (if available)
        if 'volume' in price_df.columns:
            features['volume_ma_20'] = price_df['volume'].rolling(20).mean()
            features['volume_ratio'] = price_df['volume'] / features['volume_ma_20']

        print(f"Generated {len(features.columns)} price features")

        return features

    def calculate_tweet_features(self, tweet_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features from tweet metadata and embeddings.

        Args:
            tweet_df: DataFrame with tweet data including embeddings

        Returns:
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=tweet_df.index)

        print("Calculating tweet engagement features...")
        # Engagement metrics
        features['replies_count'] = tweet_df['replies_count']
        features['reblogs_count'] = tweet_df['reblogs_count']
        features['favourites_count'] = tweet_df['favourites_count']

        # Total engagement
        features['total_engagement'] = (
            tweet_df['replies_count'] +
            tweet_df['reblogs_count'] +
            tweet_df['favourites_count']
        )

        # Engagement ratios
        features['reply_fav_ratio'] = tweet_df['replies_count'] / (tweet_df['favourites_count'] + 1)
        features['reblog_fav_ratio'] = tweet_df['reblogs_count'] / (tweet_df['favourites_count'] + 1)

        # Log-scaled engagement (handles outliers)
        features['log_total_engagement'] = np.log1p(features['total_engagement'])
        features['log_replies'] = np.log1p(tweet_df['replies_count'])
        features['log_reblogs'] = np.log1p(tweet_df['reblogs_count'])
        features['log_favourites'] = np.log1p(tweet_df['favourites_count'])

        print("Processing embeddings...")
        # Embedding features (already calculated, just extract)
        if 'embedding' in tweet_df.columns:
            embedding_array = np.vstack(tweet_df['embedding'].values)

            # Add PCA-reduced embeddings (top components)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            embedding_pca = pca.fit_transform(embedding_array)

            for i in range(50):
                features[f'embedding_pca_{i}'] = embedding_pca[:, i]

            # Keep original embeddings as well
            for i in range(embedding_array.shape[1]):
                features[f'embedding_{i}'] = embedding_array[:, i]

        print(f"Generated {len(features.columns)} tweet features")

        return features

    def create_interaction_features(self, tweet_features: pd.DataFrame, price_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between tweet and price data.

        Args:
            tweet_features: Tweet-based features
            price_features: Price-based features

        Returns:
            DataFrame with interaction features
        """
        features = pd.DataFrame(index=tweet_features.index)

        print("Creating interaction features...")

        # Engagement * Volatility interactions
        if 'vol_20' in price_features.columns and 'total_engagement' in tweet_features.columns:
            features['engagement_vol_20'] = tweet_features['total_engagement'] * price_features['vol_20']
            features['engagement_vol_5'] = tweet_features['total_engagement'] * price_features['vol_5']

        # Engagement * Momentum interactions
        if 'momentum_5' in price_features.columns:
            features['engagement_mom_5'] = tweet_features['total_engagement'] * price_features['momentum_5']
            features['engagement_mom_20'] = tweet_features['total_engagement'] * price_features['momentum_20']

        # Engagement * RSI
        if 'rsi_14' in price_features.columns:
            features['engagement_rsi'] = tweet_features['total_engagement'] * price_features['rsi_14']

        print(f"Generated {len(features.columns)} interaction features")

        return features

    def fill_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate strategies."""
        # Forward fill first (use last known value)
        features = features.ffill()

        # Backward fill for any remaining NaNs at the start
        features = features.bfill()

        # Fill any remaining with 0
        features = features.fillna(0)

        return features

    def normalize_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features to have zero mean and unit variance."""
        from sklearn.preprocessing import StandardScaler

        if fit:
            self.scaler = StandardScaler()
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)

        return pd.DataFrame(normalized, index=features.index, columns=features.columns)

if __name__ == "__main__":
    print("Feature engineering module loaded successfully")
    print("Use FeatureEngineer class to generate features for your data")
