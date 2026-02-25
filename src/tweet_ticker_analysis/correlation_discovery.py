"""
Option 1: Statistical Correlation Discovery

Computes correlation matrix between tweet embeddings/features and ticker returns.
Avoids look-ahead bias via time-based train/test split.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class CorrelationDiscovery:
    """
    Discover relationships using statistical correlation methods.
    Supports time-based train/test split to prevent look-ahead bias.
    """

    def __init__(self,
                 train_ratio: float = 0.7,
                 p_value_max: float = 0.05,
                 min_abs_correlation: float = 0.02,
                 use_price_features: bool = False):
        """
        Args:
            train_ratio: Fraction of data (by time) used for training. Backtest uses 1-train_ratio.
                        Prevents look-ahead: correlations learned only from past data.
            p_value_max: Only use features with p-value < this (statistical significance).
            min_abs_correlation: Minimum |correlation| to include a feature.
            use_price_features: If False, exclude price/technical features (ema_, vol_, rsi_, etc.)
                               to avoid potential look-ahead from price alignment.
        """
        self.correlation_matrix = None
        self.mutual_info_scores = None
        self.feature_names = None
        self.train_ratio = train_ratio
        self.p_value_max = p_value_max
        self.min_abs_correlation = min_abs_correlation
        self.use_price_features = use_price_features
        self.train_end_time = None  # Set after split; used for backtest filtering

    def discover_relationships(self, 
                              tweets_df: pd.DataFrame,
                              returns_df: pd.DataFrame,
                              tickers: List[str],
                              embedding_cols: Optional[List[str]] = None,
                              return_horizon: str = '30m') -> Dict:
        """
        Discover relationships between tweet features and ticker returns.

        Args:
            tweets_df: DataFrame with tweet data and embeddings
            returns_df: DataFrame with forward returns for each ticker
            tickers: List of ticker symbols
            embedding_cols: List of embedding column names (auto-detected if None)
            return_horizon: Return horizon to analyze (e.g., '30m', '60m')

        Returns:
            Dictionary with correlation scores per ticker-feature pair
        """
        print(f"\n{'='*80}")
        print("METHOD 1: Statistical Correlation Discovery")
        print(f"{'='*80}")

        # Auto-detect feature columns
        if embedding_cols is None:
            embedding_cols = [col for col in tweets_df.columns 
                            if col.startswith('embedding_') or col.startswith('embedding_pca_')]
        
        # Engagement features (no look-ahead - known at tweet time)
        engagement_cols = [col for col in tweets_df.columns 
                          if 'engagement' in col.lower() or col in ['replies_count', 'reblogs_count', 'favourites_count']]
        
        # Price/technical features - exclude by default to avoid look-ahead from price alignment
        price_feature_prefixes = ('ema_', 'vol_', 'rsi_', 'macd_', 'bb_', 'atr_', 
                                  'momentum_', 'dist_', 'vwap_', 'spread_')
        price_cols = [col for col in tweets_df.columns if col.startswith(price_feature_prefixes)] if self.use_price_features else []
        
        # Tweet-derived numeric features (tweet_word_count, etc. - no look-ahead)
        tweet_feature_cols = []
        for col in tweets_df.columns:
            if col.startswith('tweet_') and col not in ('tweet_id', 'tweet_content', 'tweet_url', 'tweet_time', 'tweet_idx'):
                if tweets_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    tweet_feature_cols.append(col)
        
        feature_cols = list(set(embedding_cols + engagement_cols + price_cols + tweet_feature_cols))
        self.feature_names = feature_cols

        # LOOK-AHEAD FIX: Use only training period for correlation estimation
        if 'entry_time' in tweets_df.columns:
            times = tweets_df['entry_time'].dropna()
            if len(times) > 0:
                self.train_end_time = times.quantile(self.train_ratio)
                train_tweets = tweets_df[tweets_df['entry_time'] < self.train_end_time].copy()
                print(f"  Train/test split: using {len(train_tweets)} samples (before {self.train_end_time}) for correlation")
                print(f"  Backtest will use only tweets after {self.train_end_time} (no look-ahead)")
            else:
                train_tweets = tweets_df.copy()
                self.train_end_time = None
        else:
            train_tweets = tweets_df.copy()
            self.train_end_time = None
            print("  Warning: No entry_time - using full data (potential look-ahead)")

        print(f"Analyzing {len(feature_cols)} features across {len(tickers)} tickers")
        print(f"Return horizon: {return_horizon}")
        print(f"Filters: p_value<{self.p_value_max}, |corr|>={self.min_abs_correlation}")

        results = {}

        for ticker in tickers:
            ret_col = f'{ticker}_{return_horizon}'
            
            if ret_col not in returns_df.columns:
                print(f"  Warning: {ret_col} not found, skipping {ticker}")
                continue

            # Align data (use TRAIN data only - no look-ahead)
            cols_to_merge = [c for c in feature_cols + ['tweet_id', 'entry_time'] if c in train_tweets.columns]
            aligned_df = pd.merge(
                train_tweets[cols_to_merge],
                returns_df[['tweet_id', ret_col]],
                on='tweet_id',
                how='inner'
            )

            if len(aligned_df) == 0:
                print(f"  Warning: No aligned data for {ticker}")
                continue

            # Remove NaN returns
            aligned_df = aligned_df.dropna(subset=[ret_col])
            
            if len(aligned_df) < 10:
                print(f"  Warning: Insufficient data for {ticker} ({len(aligned_df)} samples)")
                continue

            ticker_results = {}

            for feature in feature_cols:
                if feature not in aligned_df.columns:
                    continue

                feature_values = aligned_df[feature].values
                returns = aligned_df[ret_col].values

                # Remove NaN
                mask = ~(np.isnan(feature_values) | np.isnan(returns))
                if mask.sum() < 10:
                    continue

                feature_vals = feature_values[mask]
                ret_vals = returns[mask]

                # Pearson correlation
                if len(feature_vals) > 1 and feature_vals.std() > 0:
                    corr, p_value = stats.pearsonr(feature_vals, ret_vals)
                else:
                    corr, p_value = 0.0, 1.0

                # Mutual information (non-linear relationships)
                try:
                    mi_score = mutual_info_regression(
                        feature_vals.reshape(-1, 1),
                        ret_vals,
                        random_state=42
                    )[0]
                except:
                    mi_score = 0.0

                ticker_results[feature] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'mutual_info': mi_score,
                    'n_samples': len(feature_vals)
                }

            # Filter by statistical significance and minimum correlation (avoid noise)
            filtered_results = {
                f: s for f, s in ticker_results.items()
                if s['p_value'] < self.p_value_max and abs(s['correlation']) >= self.min_abs_correlation
            }

            # Fallback: if too strict, use top by p-value with relaxed min_corr
            if len(filtered_results) < 3:
                filtered_results = dict(sorted(
                    ticker_results.items(),
                    key=lambda x: (x[1]['p_value'], -abs(x[1]['correlation']))
                )[:10])
                filtered_results = {f: s for f, s in filtered_results.items() if abs(s['correlation']) >= self.min_abs_correlation * 0.5}

            # Sort by absolute correlation
            sorted_features = sorted(
                filtered_results.items(),
                key=lambda x: abs(x[1]['correlation']),
                reverse=True
            )[:20]  # Top 20

            results[ticker] = {
                'top_features': sorted_features,
                'all_features': ticker_results,
                'n_samples': len(aligned_df)
            }

            print(f"\n  {ticker}:")
            print(f"    Samples: {len(aligned_df)}")
            print(f"    Top 5 features by correlation:")
            for feat, scores in sorted_features[:5]:
                print(f"      {feat}: corr={scores['correlation']:.4f}, p={scores['p_value']:.4f}, MI={scores['mutual_info']:.4f}")

        self.correlation_matrix = results
        return results

    def get_backtest_tweets(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter tweets to backtest period only (after train_end_time).
        Prevents look-ahead: we only trade on data not used for training.
        """
        if self.train_end_time is None or 'entry_time' not in tweets_df.columns:
            return tweets_df
        return tweets_df[tweets_df['entry_time'] >= self.train_end_time].copy()

    def score_tweet_ticker(self,
                           tweet_row: pd.Series,
                           ticker: str,
                           return_horizon: str = '30m') -> Dict:
        """
        Score a single tweet-ticker pair using discovered correlations.

        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol
            return_horizon: Return horizon

        Returns:
            Dictionary with influence score and metadata
        """
        if self.correlation_matrix is None:
            raise ValueError("Must call discover_relationships() first")

        if ticker not in self.correlation_matrix:
            return {
                'influence_score': 0.0,
                'confidence': 0.0,
                'method': 'correlation',
                'matched_features': []
            }

        ticker_data = self.correlation_matrix[ticker]
        top_features = ticker_data['top_features']

        influence_score = 0.0
        matched_features = []

        for feature, scores in top_features[:10]:  # Use top 10 features
            if feature in tweet_row.index:
                feature_value = tweet_row[feature]
                # Handle duplicate columns from merge (value may be a Series)
                if isinstance(feature_value, pd.Series):
                    feature_value = feature_value.iloc[0]
                if not pd.isna(feature_value):
                    # Weight by correlation strength; scale to improve signal magnitude
                    weight = abs(scores['correlation'])
                    contribution = feature_value * scores['correlation'] * weight
                    influence_score += contribution
                    matched_features.append({
                        'feature': feature,
                        'value': float(feature_value),
                        'correlation': scores['correlation'],
                        'contribution': float(contribution)
                    })

        # Scale: sqrt(n) dilution (less aggressive than /n) to preserve signal from strong features
        if len(matched_features) > 0:
            n = len(matched_features)
            influence_score = influence_score / (n ** 0.5)
            confidence = min(1.0, n / 10.0)
        else:
            confidence = 0.0

        return {
            'influence_score': float(influence_score),
            'volatility_score': abs(influence_score) * 0.5,  # Rough estimate
            'confidence': float(confidence),
            'method': 'correlation',
            'matched_features': matched_features
        }

    def get_feature_importance(self, ticker: str) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances for a given ticker.

        Columns: feature, correlation, p_value, mutual_info, abs_correlation
        Sorted by abs_correlation descending.
        """
        if self.correlation_matrix is None or ticker not in self.correlation_matrix:
            return pd.DataFrame()

        all_features = self.correlation_matrix[ticker].get('all_features', {})
        rows = []
        for feature, scores in all_features.items():
            rows.append({
                'ticker': ticker,
                'feature': feature,
                'correlation': scores['correlation'],
                'abs_correlation': abs(scores['correlation']),
                'p_value': scores['p_value'],
                'mutual_info': scores['mutual_info'],
                'n_samples': scores['n_samples'],
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values('abs_correlation', ascending=False)
        return df.reset_index(drop=True)

    def get_all_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance for ALL tickers in a single DataFrame.
        """
        if self.correlation_matrix is None:
            return pd.DataFrame()
        frames = [self.get_feature_importance(t) for t in self.correlation_matrix]
        frames = [f for f in frames if len(f) > 0]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def print_feature_importance(self, ticker: str, top_n: int = 15):
        """
        Pretty-print the top-N most important features for a ticker.
        """
        df = self.get_feature_importance(ticker)
        if df.empty:
            print(f"  No feature importance data for {ticker}")
            return

        print(f"\n  Feature Importance — {ticker} (top {min(top_n, len(df))})")
        print(f"  {'Feature':<40} {'Corr':>8} {'p-val':>8} {'MI':>8}")
        print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in df.head(top_n).iterrows():
            sig = '*' if row['p_value'] < 0.05 else ' '
            print(f"  {row['feature']:<40} {row['correlation']:>+8.4f} {row['p_value']:>8.4f}{sig} {row['mutual_info']:>8.4f}")
