"""
Cross-asset relationship analysis and pairs trading.

Analyzes relationships between assets and implements pairs trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CrossAssetAnalyzer:
    """
    Analyze cross-asset relationships and generate pairs trading signals.
    """

    def __init__(self):
        self.asset_pairs = {}
        self.correlations = {}

    def calculate_pair_correlation(self,
                                  returns_df: pd.DataFrame,
                                  ticker1: str,
                                  ticker2: str,
                                  return_horizon: str = '30m') -> float:
        """
        Calculate correlation between two assets' returns.
        
        Args:
            returns_df: DataFrame with forward returns
            ticker1: First ticker
            ticker2: Second ticker
            return_horizon: Return horizon
            
        Returns:
            Correlation coefficient
        """
        ret_col1 = f'{ticker1}_{return_horizon}'
        ret_col2 = f'{ticker2}_{return_horizon}'
        
        if ret_col1 not in returns_df.columns or ret_col2 not in returns_df.columns:
            return 0.0
        
        # Align returns
        aligned = returns_df[[ret_col1, ret_col2]].dropna()
        
        if len(aligned) < 10:
            return 0.0
        
        corr, _ = stats.pearsonr(aligned[ret_col1], aligned[ret_col2])
        return corr

    def identify_pairs(self,
                      returns_df: pd.DataFrame,
                      tickers: List[str],
                      min_correlation: float = 0.3,
                      return_horizon: str = '30m') -> List[Tuple[str, str]]:
        """
        Identify trading pairs based on correlation.
        
        Args:
            returns_df: DataFrame with forward returns
            tickers: List of tickers
            min_correlation: Minimum correlation to consider
            return_horizon: Return horizon
            
        Returns:
            List of (ticker1, ticker2) pairs
        """
        pairs = []
        
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                corr = self.calculate_pair_correlation(returns_df, ticker1, ticker2, return_horizon)
                
                if abs(corr) > min_correlation:
                    pairs.append((ticker1, ticker2))
                    self.correlations[(ticker1, ticker2)] = corr
        
        return pairs

    def analyze_pairs_signal(self,
                            tweet_row: pd.Series,
                            scorer,
                            ticker1: str,
                            ticker2: str,
                            return_horizon: str = '30m') -> Dict:
        """
        Analyze tweet for pairs trading signal.
        
        Args:
            tweet_row: Single row from tweets DataFrame
            scorer: Scorer instance
            ticker1: First ticker in pair
            ticker2: Second ticker in pair
            return_horizon: Return horizon
            
        Returns:
            Dictionary with pairs trading signal
        """
        # Score each ticker
        try:
            score1 = scorer.score_tweet_ticker(tweet_row, ticker1, return_horizon=return_horizon)
            score2 = scorer.score_tweet_ticker(tweet_row, ticker2, return_horizon=return_horizon)
        except:
            try:
                tweet_text = tweet_row.get('tweet_content', '')
                score1 = scorer.score_tweet_ticker(tweet_text, ticker1)
                score2 = scorer.score_tweet_ticker(tweet_text, ticker2)
            except:
                return {
                    'strategy': 'none',
                    'spread': 0.0,
                    'confidence': 0.0
                }
        
        influence1 = score1.get('influence_score', 0.0)
        influence2 = score2.get('influence_score', 0.0)
        confidence1 = score1.get('confidence', 0.0)
        confidence2 = score2.get('confidence', 0.0)
        
        spread = influence1 - influence2
        avg_confidence = (confidence1 + confidence2) / 2
        
        # Generate signal
        if abs(spread) > 0.0005 and avg_confidence > 0.3:  # Minimum spread threshold
            if spread > 0:
                # ticker1 outperforms ticker2
                strategy = 'pairs_long_short'
                long_ticker = ticker1
                short_ticker = ticker2
            else:
                # ticker2 outperforms ticker1
                strategy = 'pairs_long_short'
                long_ticker = ticker2
                short_ticker = ticker1
            
            return {
                'strategy': strategy,
                'long_ticker': long_ticker,
                'short_ticker': short_ticker,
                'spread': abs(spread),
                'confidence': avg_confidence,
                'influence1': influence1,
                'influence2': influence2
            }
        else:
            return {
                'strategy': 'none',
                'spread': abs(spread),
                'confidence': avg_confidence
            }

    def get_risk_on_off_signal(self,
                               tweet_row: pd.Series,
                               scorer,
                               return_horizon: str = '30m') -> Dict:
        """
        Detect risk-on vs. risk-off signal.
        
        Risk-on: SPY/QQQ outperform TLT
        Risk-off: TLT outperforms SPY/QQQ
        
        Args:
            tweet_row: Single row from tweets DataFrame
            scorer: Scorer instance
            return_horizon: Return horizon
            
        Returns:
            Dictionary with risk-on/off signal
        """
        # Score equity vs. bonds
        try:
            spy_score = scorer.score_tweet_ticker(tweet_row, 'SPY', return_horizon=return_horizon)
            tlt_score = scorer.score_tweet_ticker(tweet_row, 'TLT', return_horizon=return_horizon)
        except:
            tweet_text = tweet_row.get('tweet_content', '')
            spy_score = scorer.score_tweet_ticker(tweet_text, 'SPY')
            tlt_score = scorer.score_tweet_ticker(tweet_text, 'TLT')
        
        spy_influence = spy_score.get('influence_score', 0.0)
        tlt_influence = tlt_score.get('influence_score', 0.0)
        
        spread = spy_influence - tlt_influence
        
        if abs(spread) > 0.0003:  # Minimum threshold
            if spread > 0:
                return {
                    'regime': 'risk_on',
                    'long': 'SPY',
                    'short': 'TLT',
                    'spread': spread,
                    'confidence': (spy_score.get('confidence', 0) + tlt_score.get('confidence', 0)) / 2
                }
            else:
                return {
                    'regime': 'risk_off',
                    'long': 'TLT',
                    'short': 'SPY',
                    'spread': abs(spread),
                    'confidence': (spy_score.get('confidence', 0) + tlt_score.get('confidence', 0)) / 2
                }
        
        return {
            'regime': 'neutral',
            'spread': abs(spread),
            'confidence': 0.0
        }
