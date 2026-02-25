"""
Advanced sentiment analysis for financial tweets.

Uses FinBERT and other financial-specific models for better sentiment detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers (optional)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class FinancialSentimentAnalyzer:
    """
    Advanced sentiment analysis using financial-specific models.
    """

    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
                - 'ProsusAI/finbert': Financial BERT (recommended)
                - 'yiyanghkust/finbert-tone': FinBERT for sentiment
                - 'distilbert-base-uncased-finetuned-sst-2-english': General sentiment
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading sentiment model: {model_name}...")
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name
                )
                print("  ✓ Model loaded successfully")
            except Exception as e:
                print(f"  Warning: Could not load {model_name}: {e}")
                print("  Falling back to simple keyword-based sentiment")
                self.pipeline = None
        else:
            print("  Warning: transformers not available, using keyword-based sentiment")

    def analyze_sentiment(self, text) -> Dict:
        """
        Analyze sentiment of text.
        
        Args:
            text: Tweet text (str)
            
        Returns:
            Dictionary with sentiment scores and metadata
        """
        if not isinstance(text, str) or len(text) < 5:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'positive_prob': 0.5,
                'negative_prob': 0.5
            }
        
        # Use transformer model if available
        if self.pipeline is not None:
            try:
                result = self.pipeline(text[:512])  # Limit length
                
                # Parse result
                label = result[0]['label'].lower()
                score = result[0]['score']
                
                # Normalize to positive/negative/neutral
                if 'positive' in label:
                    sentiment = 'positive'
                    positive_prob = score
                    negative_prob = 1 - score
                elif 'negative' in label:
                    sentiment = 'negative'
                    positive_prob = 1 - score
                    negative_prob = score
                else:
                    sentiment = 'neutral'
                    positive_prob = 0.5
                    negative_prob = 0.5
                
                # Calculate confidence
                confidence = abs(positive_prob - negative_prob)
                
                return {
                    'sentiment': sentiment,
                    'score': score if 'positive' in label else -score,
                    'confidence': confidence,
                    'positive_prob': positive_prob,
                    'negative_prob': negative_prob
                }
            except Exception as e:
                print(f"  Warning: Sentiment analysis failed: {e}")
                # Fall back to keyword-based
                return self._keyword_based_sentiment(text)
        else:
            return self._keyword_based_sentiment(text)

    def _keyword_based_sentiment(self, text: str) -> Dict:
        """Fallback keyword-based sentiment analysis."""
        text_lower = text.lower()
        
        # Positive keywords
        positive_keywords = [
            'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'win', 'victory', 'success', 'strong', 'growth', 'up',
            'bullish', 'optimistic', 'positive', 'good', 'best'
        ]
        
        # Negative keywords
        negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'worst',
            'crash', 'collapse', 'crisis', 'recession', 'down',
            'bearish', 'pessimistic', 'negative', 'weak', 'fail'
        ]
        
        # Count matches
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        # Calculate scores
        total_keywords = positive_count + negative_count
        if total_keywords > 0:
            positive_prob = positive_count / total_keywords
            negative_prob = negative_count / total_keywords
            sentiment = 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral'
            confidence = abs(positive_prob - negative_prob)
            score = (positive_count - negative_count) / max(total_keywords, 1)
        else:
            sentiment = 'neutral'
            positive_prob = 0.5
            negative_prob = 0.5
            confidence = 0.0
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'positive_prob': positive_prob,
            'negative_prob': negative_prob
        }

    def analyze_tweet_ticker_sentiment(self, tweet_text, ticker: str) -> Dict:
        """
        Analyze sentiment specifically for a ticker.
        
        This can be enhanced to use ticker-specific sentiment models
        or to weight sentiment based on ticker relevance.
        
        Args:
            tweet_text: Tweet content (str)
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker-specific sentiment
        """
        # Ensure tweet_text is a string
        if not isinstance(tweet_text, str):
            tweet_text = str(tweet_text) if tweet_text is not None else ''
        
        base_sentiment = self.analyze_sentiment(tweet_text)
        
        # Ticker-specific keyword relevance mapping (all 13 tickers)
        # Keywords represent topics that are known to move each asset
        ticker_keywords = {
            # US Equity Indices
            'SPY': [
                'market', 'economy', 'stocks', 'equities', 's&p', 'wall street',
                'gdp', 'recession', 'growth', 'bull', 'bear', 'correction'
            ],
            'DIA': [
                'dow', 'dow jones', 'industrial', 'blue chip', 'manufacturing',
                'factory', 'jobs', 'employment', 'earnings'
            ],
            # Tech / Semiconductors
            'AMD': [
                'semiconductor', 'chip', 'chips', 'amd', 'gpu', 'cpu',
                'ai', 'artificial intelligence', 'data center', 'processor'
            ],
            'NVDA': [
                'nvidia', 'gpu', 'graphics', 'ai', 'artificial intelligence',
                'data center', 'chip', 'semiconductor', 'cuda', 'training'
            ],
            'QCOM': [
                'qualcomm', '5g', 'wireless', 'mobile', 'chip', 'smartphone',
                'telecom', 'handset', 'semiconductor', 'patent'
            ],
            # Bonds / Rates
            'TLT': [
                'bonds', 'treasury', 'interest rate', 'fed', 'federal reserve',
                'rates', 'yield', 'long-term', 'duration', 'inflation',
                'quantitative easing', 'monetary policy'
            ],
            'IEF': [
                'treasury', 'bonds', '7-year', '10-year', 'interest rate',
                'yield', 'fed', 'rates', 'note'
            ],
            'SHY': [
                't-bill', 'short-term', '1-year', '2-year', '3-year',
                'treasury bill', 'money market', 'rates', 'fed funds'
            ],
            # Commodities
            'GLD': [
                'gold', 'bullion', 'inflation', 'dollar', 'currency',
                'safe haven', 'precious metal', 'uncertainty', 'war', 'crisis'
            ],
            'USO': [
                'oil', 'crude', 'opec', 'energy', 'barrel', 'petroleum',
                'gasoline', 'pipeline', 'brent', 'wti'
            ],
            # FX
            'UUP': [
                'dollar', 'usd', 'currency', 'dxy', 'forex', 'exchange rate',
                'strong dollar', 'weak dollar', 'fed', 'inflation'
            ],
            # EM / Sector ETFs
            'EWW': [
                'mexico', 'mexican', 'peso', 'nafta', 'usmca', 'nearshoring',
                'border', 'trade', 'tariff', 'latin america'
            ],
            'CYB': [
                'china', 'chinese', 'yuan', 'cny', 'renminbi', 'shanghai',
                'beijing', 'pboc', 'trade war', 'tariff', 'export', 'import'
            ],
        }

        # Sentiment direction modifiers: some tickers are INVERSE to market fear
        # e.g. TLT, GLD, SHY go UP on fear/crisis words
        fear_beneficiaries = {'TLT', 'IEF', 'SHY', 'GLD'}
        fear_keywords = ['crash', 'crisis', 'recession', 'panic', 'fear',
                         'collapse', 'uncertainty', 'war', 'default']
        is_fear_tweet = any(kw in tweet_text.lower() for kw in fear_keywords)

        # Check relevance via keyword matches
        relevance = 0.0
        if ticker in ticker_keywords:
            for keyword in ticker_keywords[ticker]:
                if keyword.lower() in tweet_text.lower():
                    relevance += 0.2

        # For flight-to-safety assets: invert sentiment on fear tweets
        adjusted_base = dict(base_sentiment)
        if ticker in fear_beneficiaries and is_fear_tweet:
            # Fear → these assets go UP, so flip sentiment direction
            adjusted_base['score'] = abs(base_sentiment['score'])
            adjusted_base['sentiment'] = 'positive'

        # Adjust confidence based on relevance
        adjusted_confidence = adjusted_base['confidence'] * (1 + relevance)
        adjusted_confidence = min(1.0, adjusted_confidence)

        return {
            **adjusted_base,
            'ticker_relevance': min(1.0, relevance),
            'adjusted_confidence': adjusted_confidence
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]


class SentimentScorer:
    """
    Wrapper that converts sentiment analysis into influence scores.
    """

    def __init__(self, sentiment_analyzer: FinancialSentimentAnalyzer):
        """
        Initialize sentiment scorer.
        
        Args:
            sentiment_analyzer: FinancialSentimentAnalyzer instance
        """
        self.analyzer = sentiment_analyzer

    def score_tweet_ticker(self, tweet_text, ticker: str) -> Dict:
        """
        Score tweet-ticker pair using sentiment.
        
        Args:
            tweet_text: Tweet content (str) or a pd.Series row from DataFrame
            ticker: Ticker symbol
            
        Returns:
            Dictionary with influence score based on sentiment
        """
        # Handle pd.Series (full row from DataFrame)
        if isinstance(tweet_text, pd.Series):
            text_val = tweet_text.get('tweet_content', '')
            if isinstance(text_val, pd.Series):
                text_val = text_val.iloc[0]
            tweet_text = str(text_val) if pd.notna(text_val) else ''
        
        sentiment_result = self.analyzer.analyze_tweet_ticker_sentiment(tweet_text, ticker)
        
        # Convert sentiment to influence score
        # Positive sentiment → positive influence
        # Negative sentiment → negative influence
        # Confidence affects magnitude
        
        base_score = sentiment_result['score']  # -1 to +1
        confidence = sentiment_result['adjusted_confidence']
        
        # Scale to expected return range
        # Typical tweet impact: -0.5% to +0.5%
        influence_score = base_score * confidence * 0.005
        
        # Volatility score: higher confidence → higher volatility impact
        volatility_score = confidence * 0.003
        
        return {
            'influence_score': influence_score,
            'volatility_score': volatility_score,
            'confidence': confidence,
            'sentiment': sentiment_result['sentiment'],
            'method': 'sentiment_analysis'
        }
