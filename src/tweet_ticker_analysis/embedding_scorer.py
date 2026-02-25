"""
Option 3: Embedding-Based Similarity Matching

Uses existing tweet embeddings to find similar historical tweets and infer relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
import warnings
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
import os

def _train_single_ticker_model(ticker: str, X: np.ndarray, y: np.ndarray, alpha: float, embedding_cols: List[str]) -> Tuple[str, Optional[Dict], Optional[Dict]]:
    """Train single ticker regression mode in parallel."""
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]

    if len(X_clean) < 20:
        return ticker, None, None

    # Train Ridge regression
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_clean, y_clean)

    # Cross-validation score
    try:
        cv_scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_mean = 0.0
        cv_std = 0.0

    # Predictions for evaluation
    y_pred = model.predict(X_clean)
    correlation = np.corrcoef(y_clean, y_pred)[0, 1] if len(y_clean) > 1 else 0.0

    model_data = {
        'model': model,
        'embedding_cols': embedding_cols,
        'cv_r2_mean': float(cv_mean),
        'cv_r2_std': float(cv_std),
        'n_samples': len(X_clean)
    }

    result_data = {
        'cv_r2_mean': float(cv_mean),
        'cv_r2_std': float(cv_std),
        'correlation': float(correlation),
        'n_samples': len(X_clean)
    }

    return ticker, model_data, result_data


class EmbeddingScorer:
    """
    Score tweet-ticker relationships using embedding similarity and regression.
    """

    def __init__(self, alpha: float = 1.0, k_neighbors: int = 50, train_ratio: float = 0.7):
        """
        Initialize embedding scorer.

        Args:
            alpha: Ridge regression regularization strength
            k_neighbors: Number of nearest neighbors for similarity-based scoring
            train_ratio: Fraction of data (by time) for training. Prevents look-ahead bias.
        """
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.train_ratio = train_ratio
        self.train_end_time = None
        self.models = {}  # {ticker: Ridge model}
        self.embeddings = None
        self.historical_data = None  # Train data only (for similarity scoring)

    def train_models(self,
                    tweets_df: pd.DataFrame,
                    returns_df: pd.DataFrame,
                    tickers: List[str],
                    return_horizon: str = '30m',
                    embedding_cols: Optional[List[str]] = None) -> Dict:
        """
        Train regression models per ticker using embeddings.

        Args:
            tweets_df: DataFrame with tweet embeddings
            returns_df: DataFrame with forward returns
            tickers: List of ticker symbols
            return_horizon: Return horizon
            embedding_cols: List of embedding column names (auto-detected if None)

        Returns:
            Dictionary with model performance metrics
        """
        print(f"\n{'='*80}")
        print("METHOD 3: Embedding-Based Regression")
        print(f"{'='*80}")

        # Auto-detect embedding columns
        if embedding_cols is None:
            # Try PCA embeddings first
            embedding_cols = [col for col in tweets_df.columns 
                            if col.startswith('embedding_pca_')]
            # If no PCA, try regular embeddings
            if len(embedding_cols) == 0:
                embedding_cols = [col for col in tweets_df.columns 
                                if col.startswith('embedding_')]
            # If still none, try tweet_embedding columns
            if len(embedding_cols) == 0:
                embedding_cols = [col for col in tweets_df.columns 
                                if 'embedding' in col.lower() and col != 'embedding']
            # Last resort: check if 'embedding' column exists (might be array/list)
            if len(embedding_cols) == 0 and 'embedding' in tweets_df.columns:
                # Try to extract from array column
                try:
                    sample_embedding = tweets_df['embedding'].iloc[0]
                    if isinstance(sample_embedding, (list, np.ndarray)):
                        # Create columns for each dimension
                        n_dims = len(sample_embedding)
                        for i in range(min(n_dims, 50)):  # Limit to 50 dimensions
                            tweets_df[f'embedding_{i}'] = tweets_df['embedding'].apply(
                                lambda x: x[i] if isinstance(x, (list, np.ndarray)) and len(x) > i else 0.0
                            )
                        embedding_cols = [f'embedding_{i}' for i in range(min(n_dims, 50))]
                        print(f"  Extracted {len(embedding_cols)} embedding dimensions from array column")
                except:
                    pass
        
        if len(embedding_cols) == 0:
            print("  Warning: No embedding columns found, skipping embedding method")
            return {}

        print(f"Using {len(embedding_cols)} embedding features")

        # LOOK-AHEAD FIX: Use only training period
        if 'entry_time' in tweets_df.columns:
            times = tweets_df['entry_time'].dropna()
            if len(times) > 0:
                self.train_end_time = times.quantile(self.train_ratio)
                train_tweets = tweets_df[tweets_df['entry_time'] < self.train_end_time].copy()
                print(f"  Train/test split: using {len(train_tweets)} samples (before {self.train_end_time})")
            else:
                train_tweets = tweets_df.copy()
                self.train_end_time = None
        else:
            train_tweets = tweets_df.copy()
            self.train_end_time = None
            print("  Warning: No entry_time - using full data (potential look-ahead)")

        # Extract embeddings (train data only - historical_data used for similarity k-NN)
        self.embeddings = train_tweets[embedding_cols].values
        self.historical_data = train_tweets.copy()

        results = {}

        # Prepare data for parallel execution
        ticker_data = []
        for ticker in tickers:
            ret_col = f'{ticker}_{return_horizon}'
            
            if ret_col not in returns_df.columns:
                print(f"  Warning: {ret_col} not found, skipping {ticker}")
                continue

            # Align data (train tweets only)
            cols = [c for c in embedding_cols + ['tweet_id', 'entry_time'] if c in train_tweets.columns]
            aligned_df = pd.merge(
                train_tweets[cols],
                returns_df[['tweet_id', ret_col]],
                on='tweet_id',
                how='inner'
            )

            if len(aligned_df) == 0:
                continue

            aligned_df = aligned_df.dropna(subset=[ret_col])
            
            if len(aligned_df) < 20:
                print(f"  Warning: Insufficient data for {ticker} ({len(aligned_df)} samples)")
                continue

            # Prepare features and targets
            X = aligned_df[embedding_cols].values
            y = aligned_df[ret_col].values
            ticker_data.append((ticker, X, y))
            
        print(f"  Training Ridge regression models for {len(ticker_data)} tickers in parallel...")
        n_jobs = min(os.cpu_count() or 4, 32)
        
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(_train_single_ticker_model)(ticker, X, y, self.alpha, embedding_cols)
            for ticker, X, y in ticker_data
        )

        for ticker, model_data, result_data in parallel_results:
            if model_data is not None:
                self.models[ticker] = model_data
                results[ticker] = result_data
                
                print(f"\n  {ticker}:")
                print(f"    Samples: {result_data['n_samples']}")
                print(f"    CV R²: {result_data['cv_r2_mean']:.4f} ± {result_data['cv_r2_std']:.4f}")
                print(f"    Correlation: {result_data['correlation']:.4f}")

        return results

    def get_backtest_tweets(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """Filter tweets to backtest period only (after train_end_time). Prevents look-ahead."""
        if self.train_end_time is None or 'entry_time' not in tweets_df.columns:
            return tweets_df
        return tweets_df[tweets_df['entry_time'] >= self.train_end_time].copy()

    def score_tweet_ticker_regression(self,
                                     tweet_row: pd.Series,
                                     ticker: str) -> Dict:
        """
        Score using trained regression model.

        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol

        Returns:
            Dictionary with influence score
        """
        if ticker not in self.models:
            return {
                'influence_score': 0.0,
                'confidence': 0.0,
                'method': 'embedding_regression'
            }

        model_data = self.models[ticker]
        model = model_data['model']
        embedding_cols = model_data['embedding_cols']

        # Extract embedding features
        embedding_values = []
        for col in embedding_cols:
            if col in tweet_row.index:
                val = tweet_row[col]
                # Handle duplicate columns from merge (value may be a Series)
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                embedding_values.append(val if not pd.isna(val) else 0.0)
            else:
                embedding_values.append(0.0)

        X = np.array([embedding_values])

        # Predict
        influence_score = model.predict(X)[0]
        confidence = min(1.0, model_data['cv_r2_mean'] + 0.5)  # Based on CV performance

        return {
            'influence_score': float(influence_score),
            'volatility_score': abs(influence_score) * 0.5,
            'confidence': float(confidence),
            'method': 'embedding_regression'
        }

    def score_tweet_ticker_similarity(self,
                                     tweet_row: pd.Series,
                                     ticker: str,
                                     returns_df: pd.DataFrame,
                                     return_horizon: str = '30m') -> Dict:
        """
        Score using similarity to historical tweets.

        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol
            returns_df: DataFrame with forward returns
            return_horizon: Return horizon

        Returns:
            Dictionary with influence score
        """
        if self.embeddings is None or self.historical_data is None:
            raise ValueError("Must call train_models() first")

        ret_col = f'{ticker}_{return_horizon}'
        if ret_col not in returns_df.columns:
            return {
                'influence_score': 0.0,
                'confidence': 0.0,
                'method': 'embedding_similarity'
            }

        # Get embedding for this tweet
        embedding_cols = [col for col in self.historical_data.columns 
                         if col.startswith('embedding_pca_') or col.startswith('embedding_')]
        embedding_cols = [col for col in embedding_cols if col.startswith('embedding_pca_')]
        if len(embedding_cols) == 0:
            embedding_cols = [col for col in self.historical_data.columns 
                            if col.startswith('embedding_')]

        tweet_embedding = []
        for col in embedding_cols[:50]:  # Use first 50 components
            if col in tweet_row.index:
                val = tweet_row[col]
                tweet_embedding.append(val if not pd.isna(val) else 0.0)
            else:
                tweet_embedding.append(0.0)

        tweet_embedding = np.array([tweet_embedding])

        # Find similar historical tweets
        historical_embeddings = self.historical_data[embedding_cols[:50]].values
        
        # Compute similarities
        similarities = cosine_similarity(tweet_embedding, historical_embeddings)[0]

        # Get top k similar
        top_k_idx = np.argsort(similarities)[::-1][:self.k_neighbors]
        top_k_similarities = similarities[top_k_idx]

        # Get corresponding returns
        historical_tweet_ids = self.historical_data.iloc[top_k_idx]['tweet_id'].values
        historical_returns = returns_df[returns_df['tweet_id'].isin(historical_tweet_ids)][ret_col].values

        if len(historical_returns) == 0:
            return {
                'influence_score': 0.0,
                'confidence': 0.0,
                'method': 'embedding_similarity'
            }

        # Weight by similarity
        # Match similarities to returns (simplified - assumes order matches)
        if len(historical_returns) == len(top_k_similarities):
            weights = top_k_similarities
            weights = weights / (weights.sum() + 1e-10)  # Normalize
            weighted_return = np.average(historical_returns, weights=weights)
        else:
            weighted_return = historical_returns.mean()

        max_similarity = top_k_similarities.max() if len(top_k_similarities) > 0 else 0.0

        return {
            'influence_score': float(weighted_return),
            'volatility_score': float(np.std(historical_returns)) if len(historical_returns) > 1 else 0.0,
            'confidence': float(max_similarity),
            'method': 'embedding_similarity',
            'n_similar': len(historical_returns)
        }

    def score_tweet_ticker(self,
                          tweet_row: pd.Series,
                          ticker: str,
                          returns_df: Optional[pd.DataFrame] = None,
                          return_horizon: str = '30m',
                          use_similarity: bool = False) -> Dict:
        """
        Score tweet-ticker pair (wrapper method).

        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol
            returns_df: Optional returns DataFrame (needed for similarity method)
            return_horizon: Return horizon
            use_similarity: Whether to use similarity-based scoring

        Returns:
            Dictionary with influence score
        """
        if use_similarity and returns_df is not None:
            return self.score_tweet_ticker_similarity(tweet_row, ticker, returns_df, return_horizon)
        else:
            return self.score_tweet_ticker_regression(tweet_row, ticker)
