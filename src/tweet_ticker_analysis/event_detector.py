"""
Event Detection & Clustering

Detects and clusters related tweets to treat event clusters as single signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class EventDetector:
    """
    Detect and cluster related tweets into events.
    """
    
    def __init__(self,
                 time_window_minutes: int = 60,
                 min_cluster_size: int = 2,
                 similarity_threshold: float = 0.3,
                 use_topic_modeling: bool = True):
        """
        Initialize event detector.
        
        Args:
            time_window_minutes: Maximum time window for tweets to be in same event
            min_cluster_size: Minimum tweets per event cluster
            similarity_threshold: Minimum similarity for tweets to be clustered
            use_topic_modeling: Whether to use topic modeling for clustering
        """
        self.time_window_minutes = time_window_minutes
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.use_topic_modeling = use_topic_modeling
        
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
    
    def detect_events(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect events by clustering related tweets.
        
        Args:
            tweets_df: DataFrame with tweet data
            
        Returns:
            DataFrame with event_id column added
        """
        print(f"\nDetecting events in {len(tweets_df)} tweets...")
        
        # Ensure entry_time is datetime
        tweets_df = tweets_df.copy()
        tweets_df['entry_time'] = pd.to_datetime(tweets_df['entry_time'])
        tweets_df = tweets_df.sort_values('entry_time').reset_index(drop=True)
        
        # Initialize event_id column
        tweets_df['event_id'] = -1
        tweets_df['event_strength'] = 1.0
        
        # Extract text features
        tweet_texts = tweets_df['tweet_content'].fillna('').astype(str).tolist()
        
        if len(tweet_texts) == 0:
            return tweets_df
        
        try:
            # Vectorize tweets
            if self.use_topic_modeling:
                tfidf_matrix = self.vectorizer.fit_transform(tweet_texts)
            else:
                # Simple keyword-based similarity
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer(max_features=50, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(tweet_texts)
            
            # Cluster tweets
            # Use DBSCAN for density-based clustering
            # Convert similarity threshold to distance threshold
            eps = 1.0 - self.similarity_threshold
            
            # Calculate pairwise distances
            distances = 1 - cosine_similarity(tfidf_matrix)
            
            # Use DBSCAN
            clustering = DBSCAN(
                eps=eps,
                min_samples=self.min_cluster_size,
                metric='precomputed'
            )
            
            cluster_labels = clustering.fit_predict(distances)
            
            # Assign event IDs
            tweets_df['event_id'] = cluster_labels
            
            # Calculate event strength (number of tweets in event)
            event_sizes = tweets_df.groupby('event_id').size()
            tweets_df['event_strength'] = tweets_df['event_id'].map(event_sizes).fillna(1.0)
            
            # Also consider temporal clustering
            tweets_df = self._temporal_clustering(tweets_df)
            
            num_events = tweets_df['event_id'].nunique()
            num_clustered = (tweets_df['event_id'] >= 0).sum()
            
            print(f"  Detected {num_events} events")
            print(f"  Clustered {num_clustered} tweets into events")
            print(f"  Average event size: {tweets_df[tweets_df['event_id'] >= 0]['event_strength'].mean():.1f} tweets")
            
        except Exception as e:
            print(f"  Warning: Event detection failed: {e}")
            # Fallback: assign each tweet its own event
            tweets_df['event_id'] = range(len(tweets_df))
            tweets_df['event_strength'] = 1.0
        
        return tweets_df
    
    def _temporal_clustering(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Refine clusters by temporal proximity.
        
        Args:
            tweets_df: DataFrame with initial event_id
            
        Returns:
            DataFrame with refined event_id
        """
        tweets_df = tweets_df.copy()
        
        # Group by existing event_id
        for event_id in tweets_df['event_id'].unique():
            if event_id < 0:  # Skip noise points
                continue
            
            event_tweets = tweets_df[tweets_df['event_id'] == event_id].copy()
            
            if len(event_tweets) < 2:
                continue
            
            # Check if tweets are within time window
            event_tweets = event_tweets.sort_values('entry_time')
            time_diffs = event_tweets['entry_time'].diff().dt.total_seconds() / 60
            
            # Split events that are too far apart in time
            split_points = time_diffs[time_diffs > self.time_window_minutes].index.tolist()
            
            if split_points:
                # Create new event IDs for split events
                current_event_id = event_id
                for split_idx in split_points:
                    # Assign new event_id to tweets after split point
                    mask = event_tweets.index >= split_idx
                    tweets_df.loc[mask & (tweets_df['event_id'] == event_id), 'event_id'] = current_event_id + 10000
                    current_event_id += 10000
        
        return tweets_df
    
    def score_event(self, event_tweets: pd.DataFrame, scorer, ticker: str, 
                   return_horizon: str = '30m') -> Dict:
        """
        Score an event (cluster of tweets) for a ticker.
        
        Args:
            event_tweets: DataFrame with tweets in the event
            scorer: Scorer instance
            ticker: Ticker symbol
            return_horizon: Return horizon
            
        Returns:
            Dictionary with event score
        """
        if len(event_tweets) == 0:
            return {
                'influence_score': 0.0,
                'confidence': 0.0,
                'event_strength': 0.0
            }
        
        # Score each tweet in event
        tweet_scores = []
        confidences = []
        
        for _, tweet_row in event_tweets.iterrows():
            try:
                if hasattr(scorer, 'score_tweet_ticker'):
                    score_dict = scorer.score_tweet_ticker(tweet_row, ticker, return_horizon=return_horizon)
                else:
                    tweet_text = tweet_row.get('tweet_content', '')
                    score_dict = scorer.score_tweet_ticker(tweet_text, ticker)
                
                tweet_scores.append(score_dict.get('influence_score', 0.0))
                confidences.append(score_dict.get('confidence', 0.0))
            except:
                tweet_scores.append(0.0)
                confidences.append(0.0)
        
        # Aggregate scores
        # Weight by recency and importance
        event_strength = len(event_tweets)
        
        # Weight recent tweets more heavily
        weights = np.linspace(0.5, 1.0, len(tweet_scores))
        weighted_scores = np.array(tweet_scores) * weights
        weighted_confidences = np.array(confidences) * weights
        
        # Aggregate: use mean of weighted scores, but boost for event strength
        influence_score = np.mean(weighted_scores) * (1.0 + 0.1 * (event_strength - 1))
        confidence = np.mean(weighted_confidences)
        
        # Boost confidence for larger events
        confidence = min(1.0, confidence * (1.0 + 0.05 * (event_strength - 1)))
        
        return {
            'influence_score': float(influence_score),
            'confidence': float(confidence),
            'event_strength': float(event_strength),
            'n_tweets': len(event_tweets),
            'tweet_scores': tweet_scores
        }
    
    def get_event_types(self, tweets_df: pd.DataFrame) -> Dict[int, str]:
        """
        Classify event types based on keywords.
        
        Args:
            tweets_df: DataFrame with event_id
            
        Returns:
            Dictionary mapping event_id to event type
        """
        event_types = {}
        
        # Define keyword patterns for event types
        event_keywords = {
            'trade_policy': ['tariff', 'trade', 'china', 'import', 'export', 'deal'],
            'economic': ['jobs', 'unemployment', 'inflation', 'gdp', 'economy', 'fed', 'interest rate'],
            'political': ['election', 'vote', 'congress', 'senate', 'bill', 'law'],
            'market': ['stock', 'market', 'dow', 'nasdaq', 's&p', 'crash', 'rally'],
            'general': []
        }
        
        for event_id in tweets_df['event_id'].unique():
            if event_id < 0:
                event_types[event_id] = 'noise'
                continue
            
            event_tweets = tweets_df[tweets_df['event_id'] == event_id]
            all_text = ' '.join(event_tweets['tweet_content'].fillna('').astype(str).str.lower())
            
            # Find matching event type
            event_type = 'general'
            max_matches = 0
            
            for etype, keywords in event_keywords.items():
                if etype == 'general':
                    continue
                
                matches = sum(1 for kw in keywords if kw in all_text)
                if matches > max_matches:
                    max_matches = matches
                    event_type = etype
            
            event_types[event_id] = event_type
        
        return event_types
