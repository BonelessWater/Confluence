"""
Multi-timeframe analysis for tweet-ticker relationships.

Scores tweets across multiple timeframes and selects optimal horizon.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiTimeframeScorer:
    """
    Score tweets across multiple timeframes and select optimal horizon.
    """

    def __init__(self, base_scorer, horizons: List[int] = [5, 15, 30, 60, 240]):
        """
        Initialize multi-timeframe scorer.
        
        Args:
            base_scorer: Base scorer (CorrelationDiscovery, BagOfWordsScorer, etc.)
            horizons: List of timeframes in minutes
        """
        self.base_scorer = base_scorer
        self.horizons = horizons
        self.horizon_performance = {}  # Track performance per horizon

    def score_all_timeframes(self,
                            tweet_row: pd.Series,
                            ticker: str,
                            returns_df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Score tweet-ticker pair across all timeframes.
        
        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol
            returns_df: DataFrame with forward returns
            
        Returns:
            Dictionary mapping horizon to score dictionary
        """
        scores = {}
        
        for horizon in self.horizons:
            ret_col = f'{ticker}_{horizon}m'
            if ret_col not in returns_df.columns:
                continue
            
            return_horizon = f'{horizon}m'
            
            try:
                # Score for this horizon
                if hasattr(self.base_scorer, 'score_tweet_ticker'):
                    score = self.base_scorer.score_tweet_ticker(
                        tweet_row, ticker, return_horizon=return_horizon
                    )
                else:
                    # Try alternative method signatures
                    tweet_text = tweet_row.get('tweet_content', '')
                    if tweet_text:
                        score = self.base_scorer.score_tweet_ticker(tweet_text, ticker)
                    else:
                        score = {'influence_score': 0.0, 'confidence': 0.0}
                
                scores[horizon] = score
            except Exception as e:
                scores[horizon] = {'influence_score': 0.0, 'confidence': 0.0}
        
        return scores

    def select_optimal_horizon(self, scores: Dict[int, Dict]) -> Tuple[int, Dict]:
        """
        Select optimal timeframe based on scores.
        
        Strategy:
        - Prefer horizons with higher absolute influence scores
        - Weight by confidence
        - Consider historical performance if available
        
        Args:
            scores: Dictionary of horizon -> score dict
            
        Returns:
            Tuple of (optimal_horizon, score_dict)
        """
        if not scores:
            return None, {'influence_score': 0.0, 'confidence': 0.0}
        
        # Score each horizon
        horizon_scores = {}
        for horizon, score_dict in scores.items():
            influence = abs(score_dict.get('influence_score', 0.0))
            confidence = score_dict.get('confidence', 0.0)
            
            # Combined score: influence * confidence
            combined_score = influence * (0.7 + 0.3 * confidence)
            
            # Adjust by historical performance if available
            if horizon in self.horizon_performance:
                perf_multiplier = 1.0 + self.horizon_performance[horizon]
                combined_score *= perf_multiplier
            
            horizon_scores[horizon] = combined_score
        
        # Select best horizon
        optimal_horizon = max(horizon_scores.items(), key=lambda x: x[1])[0]
        optimal_score = scores[optimal_horizon]
        
        return optimal_horizon, optimal_score

    def score_tweet_ticker(self,
                          tweet_row: pd.Series,
                          ticker: str,
                          returns_df: pd.DataFrame) -> Dict:
        """
        Score tweet-ticker pair using optimal timeframe.
        
        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol
            returns_df: DataFrame with forward returns
            
        Returns:
            Dictionary with influence score and optimal horizon
        """
        # Score all timeframes
        all_scores = self.score_all_timeframes(tweet_row, ticker, returns_df)
        
        # Select optimal
        optimal_horizon, optimal_score = self.select_optimal_horizon(all_scores)
        
        return {
            **optimal_score,
            'optimal_horizon': optimal_horizon,
            'all_horizon_scores': {h: s.get('influence_score', 0.0) for h, s in all_scores.items()}
        }

    def update_performance(self, horizon: int, actual_return: float):
        """
        Update historical performance for a horizon.
        
        Args:
            horizon: Timeframe in minutes
            actual_return: Actual return achieved
        """
        if horizon not in self.horizon_performance:
            self.horizon_performance[horizon] = []
        
        self.horizon_performance[horizon].append(actual_return)
        
        # Keep only recent performance (last 50)
        if len(self.horizon_performance[horizon]) > 50:
            self.horizon_performance[horizon] = self.horizon_performance[horizon][-50:]
        
        # Calculate average performance
        avg_perf = np.mean(self.horizon_performance[horizon])
        self.horizon_performance[horizon] = avg_perf
