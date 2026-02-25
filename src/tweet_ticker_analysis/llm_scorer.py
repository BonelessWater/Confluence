"""
Option 4: LLM-Based Semantic Analysis

Uses Large Language Models to analyze tweet content and rate ticker-specific influence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import OpenAI (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMScorer:
    """
    Score tweet-ticker relationships using LLM semantic analysis.
    
    Note: Requires OpenAI API key in environment or .env file.
    Set OPENAI_API_KEY environment variable or use .env file.
    """

    def __init__(self, model: str = 'gpt-3.5-turbo', use_llm: bool = True):
        """
        Initialize LLM scorer.

        Args:
            model: OpenAI model to use ('gpt-3.5-turbo' or 'gpt-4')
            use_llm: Whether to actually use LLM (False for testing without API)
        """
        self.model = model
        self.use_llm = use_llm and OPENAI_AVAILABLE
        
        if self.use_llm:
            try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
                if not openai.api_key:
                    print("Warning: OPENAI_API_KEY not found. LLM scoring will be disabled.")
                    self.use_llm = False
            except:
                self.use_llm = False
        
        if not self.use_llm:
            print("LLM scoring disabled (no API key or OpenAI not available)")
            print("  Set OPENAI_API_KEY environment variable to enable")

    def score_tweet_ticker(self,
                          tweet_text,
                          ticker: str,
                          ticker_description: Optional[str] = None) -> Dict:
        """
        Score a single tweet-ticker pair using LLM.

        Args:
            tweet_text: Tweet content (str) or a pd.Series row from DataFrame
            ticker: Ticker symbol
            ticker_description: Optional description of ticker (e.g., "S&P 500 ETF")

        Returns:
            Dictionary with influence score and metadata
        """
        # Handle pd.Series (full row from DataFrame)
        if isinstance(tweet_text, pd.Series):
            text_val = tweet_text.get('tweet_content', '')
            if isinstance(text_val, pd.Series):
                text_val = text_val.iloc[0]
            tweet_text = str(text_val) if pd.notna(text_val) else ''
        
        if not self.use_llm:
            # Return zero scores if LLM not available
            return {
                'influence_score': 0.0,
                'volatility_score': 0.0,
                'confidence': 0.0,
                'method': 'llm_disabled',
                'reasoning': 'LLM API not available'
            }

        # Default ticker descriptions
        ticker_descriptions = {
            'SPY': 'S&P 500 ETF (broad US equity market)',
            'QQQ': 'NASDAQ-100 ETF (tech-heavy US equity)',
            'DIA': 'Dow Jones Industrial Average ETF',
            'IWM': 'Russell 2000 ETF (small-cap US equity)',
            'TLT': '20+ Year Treasury ETF (long-term bonds)',
            'GLD': 'Gold ETF',
            'USO': 'US Oil ETF',
            'EWW': 'iShares MSCI Mexico ETF',
            'CYB': 'China Yuan ETF',
            'UUP': 'US Dollar Index ETF',
            'AMD': 'Advanced Micro Devices (tech stock)',
            'NVDA': 'NVIDIA Corporation (tech stock)',
            'QCOM': 'Qualcomm Inc (tech stock)'
        }

        description = ticker_description or ticker_descriptions.get(ticker, ticker)

        prompt = f"""Analyze this tweet and rate its potential impact on {ticker} ({description}).

Tweet: "{tweet_text}"

Provide a JSON response with:
1. influence_score: Expected return impact (-1 to +1, where +1 is very bullish, -1 is very bearish)
2. volatility_score: Expected volatility increase (0 to 1, where 1 is high volatility)
3. confidence: How certain is this assessment (0 to 1)
4. reasoning: Brief explanation (max 50 words)

Format as valid JSON only:
{{
    "influence_score": <float>,
    "volatility_score": <float>,
    "confidence": <float>,
    "reasoning": "<text>"
}}"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON
            if content.startswith('```'):
                # Remove markdown code blocks
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)

            return {
                'influence_score': float(result.get('influence_score', 0.0)),
                'volatility_score': float(result.get('volatility_score', 0.0)),
                'confidence': float(result.get('confidence', 0.0)),
                'method': f'llm_{self.model}',
                'reasoning': result.get('reasoning', '')
            }

        except Exception as e:
            print(f"  Warning: LLM scoring failed for {ticker}: {e}")
            return {
                'influence_score': 0.0,
                'volatility_score': 0.0,
                'confidence': 0.0,
                'method': 'llm_error',
                'reasoning': str(e)
            }

    def batch_score(self,
                   tweets_df: pd.DataFrame,
                   tickers: List[str],
                   ticker_descriptions: Optional[Dict[str, str]] = None,
                   max_tweets: Optional[int] = None) -> pd.DataFrame:
        """
        Batch score multiple tweets (with rate limiting).

        Args:
            tweets_df: DataFrame with tweet content
            tickers: List of tickers to score
            ticker_descriptions: Optional dict of ticker descriptions
            max_tweets: Maximum number of tweets to score (for cost control)

        Returns:
            DataFrame with scores for each tweet-ticker pair
        """
        if not self.use_llm:
            print("LLM scoring disabled, returning empty results")
            return pd.DataFrame()

        if 'tweet_content' not in tweets_df.columns:
            raise ValueError("tweets_df must contain 'tweet_content' column")

        results = []
        tweets_to_score = tweets_df.head(max_tweets) if max_tweets else tweets_df

        print(f"\nScoring {len(tweets_to_score)} tweets × {len(tickers)} tickers = {len(tweets_to_score) * len(tickers)} API calls")
        print("  (This may take a while and incur costs)")

        for idx, row in tweets_to_score.iterrows():
            tweet_text = row['tweet_content']
            if pd.isna(tweet_text) or len(str(tweet_text)) < 5:
                continue

            for ticker in tickers:
                desc = ticker_descriptions.get(ticker) if ticker_descriptions else None
                score = self.score_tweet_ticker(tweet_text, ticker, desc)
                
                results.append({
                    'tweet_id': row.get('tweet_id', idx),
                    'ticker': ticker,
                    'influence_score': score['influence_score'],
                    'volatility_score': score['volatility_score'],
                    'confidence': score['confidence'],
                    'method': score['method'],
                    'reasoning': score.get('reasoning', '')
                })

        return pd.DataFrame(results)
