"""
Sentiment intensity scoring for tweets.

Enhances base scorers with sentiment analysis to better capture tweet tone.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import VADER sentiment analyzer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Install with: pip install vaderSentiment")
    print("  Falling back to simple sentiment scoring")


class SentimentEnhancer:
    """
    Enhance tweet-ticker scores with sentiment intensity.
    """

    def __init__(self, use_vader: bool = True):
        """
        Initialize sentiment enhancer.
        
        Args:
            use_vader: Whether to use VADER sentiment analyzer
        """
        self.use_vader = use_vader and VADER_AVAILABLE
        
        if self.use_vader:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None

    def score_sentiment_simple(self, text: str) -> float:
        """
        Simple sentiment scoring without VADER.
        
        Uses keyword-based approach.
        """
        if not isinstance(text, str):
            return 0.0
        
        text_lower = text.lower()
        
        # Positive keywords with weights
        positive_keywords = {
            'great': 0.5, 'excellent': 0.6, 'amazing': 0.5, 'wonderful': 0.4,
            'win': 0.6, 'victory': 0.5, 'success': 0.4, 'best': 0.3,
            'strong': 0.4, 'good': 0.2, 'positive': 0.3, 'up': 0.2,
            'boom': 0.5, 'surge': 0.4, 'rally': 0.4, 'soar': 0.5
        }
        
        # Negative keywords with weights
        negative_keywords = {
            'terrible': 0.5, 'awful': 0.5, 'disaster': 0.6, 'crash': 0.6,
            'fail': 0.4, 'loss': 0.3, 'worst': 0.4, 'weak': 0.3,
            'negative': 0.3, 'down': 0.2, 'collapse': 0.5, 'plunge': 0.5,
            'crisis': 0.4, 'recession': 0.5, 'depression': 0.6
        }
        
        # Modifiers
        intensifiers = {'very': 1.3, 'extremely': 1.5, 'incredibly': 1.4, 'really': 1.2}
        diminishers = {'slightly': 0.7, 'somewhat': 0.8, 'a bit': 0.8}
        
        # Negation words
        negations = ['not', "n't", 'no', 'never', 'none']
        
        words = text_lower.split()
        sentiment_score = 0.0
        
        for i, word in enumerate(words):
            # Check for negation
            is_negated = False
            if i > 0 and words[i-1] in negations:
                is_negated = True
            
            # Check for intensifier
            intensifier = 1.0
            if i > 0 and words[i-1] in intensifiers:
                intensifier = intensifiers[words[i-1]]
            elif i > 0 and words[i-1] in diminishers:
                intensifier = diminishers[words[i-1]]
            
            # Score word
            if word in positive_keywords:
                score = positive_keywords[word] * intensifier
                if is_negated:
                    score = -score * 0.5  # Negated positive becomes negative
                sentiment_score += score
            elif word in negative_keywords:
                score = negative_keywords[word] * intensifier
                if is_negated:
                    score = -score * 0.5  # Negated negative becomes positive
                sentiment_score += score
        
        # Normalize to -1 to +1 range
        sentiment_score = np.tanh(sentiment_score / 5.0)
        
        return sentiment_score

    def score_sentiment(self, text: str) -> Dict:
        """
        Score sentiment of text.
        
        Returns:
            Dictionary with sentiment scores
        """
        if pd.isna(text) or not isinstance(text, str) or len(text) < 3:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        
        if self.use_vader:
            scores = self.analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        else:
            # Use simple scoring
            compound = self.score_sentiment_simple(text)
            return {
                'compound': compound,
                'positive': max(0, compound),
                'negative': max(0, -compound),
                'neutral': 1.0 - abs(compound)
            }

    def enhance_score(self, base_score: float, tweet_text: str, 
                     sentiment_weight: float = 0.3) -> Dict:
        """
        Enhance base score with sentiment.
        
        Args:
            base_score: Base influence score from method
            tweet_text: Tweet text
            sentiment_weight: How much to weight sentiment (0-1)
            
        Returns:
            Dictionary with enhanced score and sentiment info
        """
        sentiment = self.score_sentiment(tweet_text)
        
        # Combine: Base score adjusted by sentiment
        # Positive sentiment amplifies positive scores, reduces negative
        # Negative sentiment amplifies negative scores, reduces positive
        
        if base_score >= 0:
            # Positive base score
            sentiment_multiplier = 1 + (sentiment['compound'] * sentiment_weight)
        else:
            # Negative base score
            sentiment_multiplier = 1 - (sentiment['compound'] * sentiment_weight)
        
        enhanced_score = base_score * sentiment_multiplier
        
        # Confidence boost if sentiment is strong
        confidence_boost = abs(sentiment['compound']) * 0.2
        
        return {
            'influence_score': enhanced_score,
            'base_score': base_score,
            'sentiment_compound': sentiment['compound'],
            'sentiment_positive': sentiment['positive'],
            'sentiment_negative': sentiment['negative'],
            'sentiment_multiplier': sentiment_multiplier,
            'confidence_boost': confidence_boost
        }
