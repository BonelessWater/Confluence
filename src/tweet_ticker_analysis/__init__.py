"""
Tweet-Ticker Relationship Analysis Module

This module implements multiple methods for discovering relationships between
tweets and stock tickers, and scoring the influence/volatility impact.
"""

from .correlation_discovery import CorrelationDiscovery
from .bag_of_words_scorer import BagOfWordsScorer
from .embedding_scorer import EmbeddingScorer
from .llm_scorer import LLMScorer
from .ensemble_scorer import EnsembleScorer
from .tweet_cleaner import TweetCleaner
from .correlation_validator import CorrelationValidator
from .sentiment_analyzer import FinancialSentimentAnalyzer, SentimentScorer
from .market_regime import MarketRegimeDetector, MarketRegime
from .position_sizing import PositionSizer, AdaptivePositionSizer
from .multi_timeframe_scorer import MultiTimeframeScorer
from .cross_asset_analyzer import CrossAssetAnalyzer
from .sentiment_enhancer import SentimentEnhancer
from .regime_detector import MarketRegimeDetector as SimpleMarketRegimeDetector
from .event_detector import EventDetector

__all__ = [
    'CorrelationDiscovery',
    'BagOfWordsScorer',
    'EmbeddingScorer',
    'LLMScorer',
    'EnsembleScorer',
    'TweetCleaner',
    'CorrelationValidator',
    'FinancialSentimentAnalyzer',
    'SentimentScorer',
    'MarketRegimeDetector',
    'MarketRegime',
    'PositionSizer',
    'AdaptivePositionSizer',
    'MultiTimeframeScorer',
    'CrossAssetAnalyzer',
    'SentimentEnhancer',
    'SimpleMarketRegimeDetector',
    'EventDetector'
]
