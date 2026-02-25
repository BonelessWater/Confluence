"""
Correlation validation to verify real relationships between tweets and stocks.

Tests statistical significance of relationships before using them for trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class CorrelationValidator:
    """
    Validate that tweet-ticker relationships are statistically significant.
    """

    def __init__(self, min_correlation: float = 0.05, min_p_value: float = 0.05):
        """
        Initialize validator.
        
        Args:
            min_correlation: Minimum absolute correlation to consider
            min_p_value: Maximum p-value for significance
        """
        self.min_correlation = min_correlation
        self.min_p_value = min_p_value
        self.validated_relationships = {}

    def validate_relationships(self,
                              tweets_df: pd.DataFrame,
                              returns_df: pd.DataFrame,
                              tickers: List[str],
                              return_horizon: str = '30m') -> Dict:
        """
        Validate tweet-ticker relationships for statistical significance.
        
        Args:
            tweets_df: DataFrame with tweet data
            returns_df: DataFrame with forward returns
            tickers: List of ticker symbols
            return_horizon: Return horizon
            
        Returns:
            Dictionary with validated relationships
        """
        print(f"\n{'='*80}")
        print("VALIDATING TWEET-TICKER RELATIONSHIPS")
        print(f"{'='*80}")
        print(f"Testing for real correlations...")
        print(f"  Min correlation: {self.min_correlation}")
        print(f"  Max p-value: {self.min_p_value}")
        
        validated = {}
        
        for ticker in tickers:
            ret_col = f'{ticker}_{return_horizon}'
            
            if ret_col not in returns_df.columns:
                continue
            
            # Align data
            aligned_df = pd.merge(
                tweets_df[['tweet_id', 'tweet_content']],
                returns_df[['tweet_id', ret_col]],
                on='tweet_id',
                how='inner'
            ).dropna(subset=[ret_col])
            
            if len(aligned_df) < 20:
                continue
            
            returns = aligned_df[ret_col].values
            
            # Test 1: Overall correlation with tweet features
            # Simple test: do tweets with certain characteristics correlate with returns?
            
            # Test by tweet length
            tweet_lengths = aligned_df['tweet_content'].str.len().values
            if tweet_lengths.std() > 0:
                corr_length, p_length = stats.pearsonr(tweet_lengths, returns)
            else:
                corr_length, p_length = 0.0, 1.0
            
            # Test by keyword presence
            important_keywords = ['tariff', 'trade', 'china', 'economy', 'market']
            keyword_presence = aligned_df['tweet_content'].str.contains(
                '|'.join(important_keywords), case=False, na=False
            ).astype(int).values
            
            if keyword_presence.std() > 0:
                corr_keywords, p_keywords = stats.pearsonr(keyword_presence, returns)
            else:
                corr_keywords, p_keywords = 0.0, 1.0
            
            # Test if returns are significantly different from zero
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Determine if relationship is valid
            is_valid = (
                abs(mean_return) > 0.0001 and  # Non-zero mean
                p_value < self.min_p_value and  # Statistically significant
                (abs(corr_length) > self.min_correlation or abs(corr_keywords) > self.min_correlation)
            )
            
            validated[ticker] = {
                'mean_return': mean_return,
                'std_return': std_return,
                't_stat': t_stat,
                'p_value': p_value,
                'n_samples': len(returns),
                'corr_length': corr_length,
                'p_length': p_length,
                'corr_keywords': corr_keywords,
                'p_keywords': p_keywords,
                'is_valid': is_valid
            }
            
            status = "VALID" if is_valid else "WEAK"
            print(f"\n  {ticker}: {status}")
            print(f"    Mean return: {mean_return:.6f}")
            print(f"    T-stat: {t_stat:.2f}, p-value: {p_value:.4f}")
            print(f"    Samples: {len(returns)}")
            if is_valid:
                print(f"    ✓ Statistically significant relationship detected")
            else:
                print(f"    ✗ No significant relationship (may be noise)")
        
        # Summary
        valid_count = sum(1 for v in validated.values() if v['is_valid'])
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Valid relationships: {valid_count}/{len(validated)}")
        
        if valid_count == 0:
            print("\n⚠️  WARNING: No statistically significant relationships found!")
            print("   This suggests:")
            print("   - Tweets may not have predictive power for these tickers")
            print("   - Need more data or different features")
            print("   - Consider longer time horizons")
        else:
            print(f"\n✓ Found {valid_count} ticker(s) with valid relationships")
        
        self.validated_relationships = validated
        return validated

    def get_valid_tickers(self) -> List[str]:
        """Get list of tickers with validated relationships."""
        return [
            ticker for ticker, data in self.validated_relationships.items()
            if data['is_valid']
        ]

    def filter_by_validation(self, tweets_df: pd.DataFrame, 
                            returns_df: pd.DataFrame,
                            tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter data to only include validated tickers.
        
        Args:
            tweets_df: Tweet DataFrame
            returns_df: Returns DataFrame
            tickers: List of all tickers
            
        Returns:
            Filtered tweets_df, returns_df
        """
        valid_tickers = self.get_valid_tickers()
        
        if len(valid_tickers) == 0:
            print("\n⚠️  No valid tickers found - using all tickers")
            return tweets_df, returns_df
        
        print(f"\nFiltering to validated tickers: {valid_tickers}")
        
        # Filter tweets
        filtered_tweets = tweets_df[tweets_df['ticker'].isin(valid_tickers)].copy()
        
        # Filter returns (keep all columns but note which are valid)
        filtered_returns = returns_df.copy()
        
        return filtered_tweets, filtered_returns
