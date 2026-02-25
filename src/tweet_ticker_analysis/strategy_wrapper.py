"""
Strategy wrapper that converts tweet-ticker scoring methods into trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.models.base_model import BaseTradingModel
from .correlation_discovery import CorrelationDiscovery
from .bag_of_words_scorer import BagOfWordsScorer
from .embedding_scorer import EmbeddingScorer
from .llm_scorer import LLMScorer
from .ensemble_scorer import EnsembleScorer
import warnings
warnings.filterwarnings('ignore')


class TweetTickerStrategy(BaseTradingModel):
    """
    Wrapper that converts a tweet-ticker scoring method into a trading model.
    """

    def __init__(self, name: str, scorer, tweets_df: pd.DataFrame, returns_df: pd.DataFrame):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            scorer: Scorer instance (CorrelationDiscovery, BagOfWordsScorer, etc.)
            tweets_df: DataFrame with tweet data
            returns_df: DataFrame with forward returns
        """
        super().__init__(name=name)
        self.scorer = scorer
        self.tweets_df = tweets_df
        self.returns_df = returns_df
        self.trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Training is done during scorer initialization."""
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict returns for given features.

        Note: For tweet-ticker strategies, we need to map features back to tweets
        and use the scorer. This is a simplified version.
        """
        # This is a placeholder - actual prediction happens in backtest
        return np.zeros(len(X))

    def predict_tweet_ticker(self, tweet_row: pd.Series, ticker: str, 
                            return_horizon: str = '30m') -> float:
        """
        Predict influence score for a tweet-ticker pair.

        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol
            return_horizon: Return horizon

        Returns:
            Influence score (expected return)
        """
        if isinstance(self.scorer, CorrelationDiscovery):
            score = self.scorer.score_tweet_ticker(tweet_row, ticker, return_horizon)
        elif isinstance(self.scorer, BagOfWordsScorer):
            tweet_text = tweet_row.get('tweet_content', '')
            score = self.scorer.score_tweet_ticker(tweet_text, ticker)
        elif isinstance(self.scorer, EmbeddingScorer):
            score = self.scorer.score_tweet_ticker(tweet_row, ticker, self.returns_df, return_horizon)
        elif isinstance(self.scorer, LLMScorer):
            tweet_text = tweet_row.get('tweet_content', '')
            score = self.scorer.score_tweet_ticker(tweet_text, ticker)
        elif isinstance(self.scorer, EnsembleScorer):
            score = self.scorer.score_tweet_ticker(tweet_row, ticker, self.returns_df, return_horizon)
        else:
            return 0.0

        return score.get('influence_score', 0.0)
