"""
Tweet cleaning and filtering utilities.

Cleans HTML tags, filters low-quality tweets, and ranks tweets by importance.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class TweetCleaner:
    """
    Clean and filter tweets for better analysis.
    """

    def __init__(self):
        self.cleaned_tweets = None

    def clean_html_tags(self, text: str) -> str:
        """
        Remove HTML tags and entities from tweet text.
        
        Args:
            text: Raw tweet text with HTML
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#39;', "'", text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def clean_tweets(self, tweets_df: pd.DataFrame, 
                    text_column: str = 'tweet_content') -> pd.DataFrame:
        """
        Clean all tweets in DataFrame.
        
        Args:
            tweets_df: DataFrame with tweet data
            text_column: Name of column containing tweet text
            
        Returns:
            DataFrame with cleaned tweets
        """
        print(f"\nCleaning tweets...")
        print(f"  Original tweets: {len(tweets_df)}")
        
        tweets_df = tweets_df.copy()
        
        # Clean text
        if text_column in tweets_df.columns:
            tweets_df['tweet_content_original'] = tweets_df[text_column].copy()
            tweets_df[text_column] = tweets_df[text_column].apply(self.clean_html_tags)
        
        # Remove empty tweets
        if text_column in tweets_df.columns:
            before = len(tweets_df)
            tweets_df = tweets_df[tweets_df[text_column].str.len() > 5].copy()
            after = len(tweets_df)
            print(f"  Removed {before - after} empty/short tweets")
        
        # Remove duplicates — include ticker in subset so each (tweet, ticker) pair is kept.
        # The dataset has one row per (tweet_id, ticker); deduping on tweet_id alone would
        # drop all non-first tickers (typically everything except AMD which is first in concat).
        if 'tweet_id' in tweets_df.columns:
            before = len(tweets_df)
            dedup_cols = ['tweet_id', 'ticker'] if 'ticker' in tweets_df.columns else ['tweet_id']
            tweets_df = tweets_df.drop_duplicates(subset=dedup_cols, keep='first')
            after = len(tweets_df)
            print(f"  Removed {before - after} duplicate tweets")
        elif text_column in tweets_df.columns:
            before = len(tweets_df)
            dedup_cols = [text_column, 'ticker'] if 'ticker' in tweets_df.columns else [text_column]
            tweets_df = tweets_df.drop_duplicates(subset=dedup_cols, keep='first')
            after = len(tweets_df)
            print(f"  Removed {before - after} duplicate tweets")
        
        print(f"  Cleaned tweets: {len(tweets_df)}")
        
        self.cleaned_tweets = tweets_df
        return tweets_df

    def calculate_tweet_importance_score(self, tweets_df: pd.DataFrame) -> pd.Series:
        """
        Calculate importance score for each tweet.
        
        Factors:
        - Length (longer tweets often more substantive)
        - Engagement metrics (replies, reblogs, favourites)
        - Keyword density (mentions of important terms)
        - Uniqueness (less repetitive)
        
        Args:
            tweets_df: DataFrame with tweet data
            
        Returns:
            Series with importance scores
        """
        scores = pd.Series(0.0, index=tweets_df.index)
        
        # Length score (normalized)
        if 'tweet_content' in tweets_df.columns:
            lengths = tweets_df['tweet_content'].str.len()
            if lengths.max() > 0:
                scores += (lengths / lengths.max()) * 0.2
        
        # Engagement score
        engagement_cols = ['replies_count', 'reblogs_count', 'favourites_count']
        engagement_scores = []
        for col in engagement_cols:
            if col in tweets_df.columns:
                engagement_scores.append(tweets_df[col].fillna(0))
        
        if engagement_scores:
            total_engagement = pd.concat(engagement_scores, axis=1).sum(axis=1)
            if total_engagement.max() > 0:
                scores += (total_engagement / total_engagement.max()) * 0.4
        
        # Keyword importance (financial/political keywords)
        important_keywords = [
            'tariff', 'trade', 'china', 'mexico', 'economy', 'market', 'stock',
            'inflation', 'fed', 'interest', 'rate', 'dollar', 'currency',
            'tax', 'policy', 'regulation', 'deal', 'agreement', 'war',
            'crisis', 'recession', 'growth', 'jobs', 'unemployment'
        ]
        
        if 'tweet_content' in tweets_df.columns:
            keyword_counts = pd.Series(0, index=tweets_df.index)
            for keyword in important_keywords:
                keyword_counts += tweets_df['tweet_content'].str.contains(
                    keyword, case=False, na=False
                ).astype(int)
            
            if keyword_counts.max() > 0:
                scores += (keyword_counts / keyword_counts.max()) * 0.3
        
        # Uniqueness score (inverse of similarity to other tweets)
        # Simple: penalize very short tweets
        if 'tweet_content' in tweets_df.columns:
            uniqueness = tweets_df['tweet_content'].str.len()
            if uniqueness.max() > 0:
                scores += (uniqueness / uniqueness.max()) * 0.1
        
        return scores

    def filter_important_tweets(self, tweets_df: pd.DataFrame,
                               top_percentile: float = 0.3,
                               min_score: Optional[float] = None) -> pd.DataFrame:
        """
        Filter tweets to keep only the most important ones.
        
        Args:
            tweets_df: DataFrame with tweet data
            top_percentile: Keep top X percentile (0.3 = top 30%)
            min_score: Minimum importance score (overrides percentile if set)
            
        Returns:
            Filtered DataFrame
        """
        print(f"\nFiltering tweets by importance...")
        print(f"  Original tweets: {len(tweets_df)}")
        
        # Calculate importance scores
        importance_scores = self.calculate_tweet_importance_score(tweets_df)
        tweets_df['importance_score'] = importance_scores
        
        # Filter
        if min_score is not None:
            filtered = tweets_df[tweets_df['importance_score'] >= min_score].copy()
        else:
            threshold = importance_scores.quantile(1 - top_percentile)
            filtered = tweets_df[tweets_df['importance_score'] >= threshold].copy()
        
        print(f"  Filtered tweets: {len(filtered)} ({len(filtered)/len(tweets_df)*100:.1f}%)")
        print(f"  Importance score range: {filtered['importance_score'].min():.4f} - {filtered['importance_score'].max():.4f}")
        
        return filtered.sort_values('importance_score', ascending=False).reset_index(drop=True)
