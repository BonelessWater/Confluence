"""
Option 5: Hybrid Ensemble Approach

Combines multiple methods with learned weights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .correlation_discovery import CorrelationDiscovery
from .bag_of_words_scorer import BagOfWordsScorer
from .embedding_scorer import EmbeddingScorer
from .llm_scorer import LLMScorer
import warnings
warnings.filterwarnings('ignore')


class EnsembleScorer:
    """
    Ensemble scorer combining multiple methods.
    """

    def __init__(self,
                 use_correlation: bool = True,
                 use_bow: bool = True,
                 use_embedding: bool = True,
                 use_llm: bool = False):
        """
        Initialize ensemble scorer.

        Args:
            use_correlation: Whether to use correlation method
            use_bow: Whether to use bag-of-words method
            use_embedding: Whether to use embedding method
            use_llm: Whether to use LLM method (requires API key)
        """
        self.use_correlation = use_correlation
        self.use_bow = use_bow
        self.use_embedding = use_embedding
        self.use_llm = use_llm

        # Initialize scorers (all use train/test split to avoid look-ahead)
        self.correlation_scorer = CorrelationDiscovery(
            train_ratio=0.7, p_value_max=0.05, min_abs_correlation=0.02, use_price_features=False
        ) if use_correlation else None
        self.bow_scorer = BagOfWordsScorer(train_ratio=0.7) if use_bow else None
        self.embedding_scorer = EmbeddingScorer(train_ratio=0.7) if use_embedding else None
        self.llm_scorer = LLMScorer(use_llm=use_llm) if use_llm else None

        # Stacking meta-model (trained later via learn_weights if desired)
        self.use_stacking = False
        self.meta_model = None

        # Performance tracking for dynamic weights (must be initialized)
        self.method_performance = {
            'correlation': [],
            'bag_of_words': [],
            'embedding': [],
            'llm': []
        }
        self.performance_window = 100

        # Default weights - favor embedding (best performer), reduce BOW (underperforms)
        self.weights = {
            'correlation': 0.2,
            'bag_of_words': 0.2,   # Reduced from 0.3 - BOW tends to underperform
            'embedding': 0.6,       # Increased - typically best performer
            'llm': 0.2
        }

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def train_all_methods(self,
                         tweets_df: pd.DataFrame,
                         returns_df: pd.DataFrame,
                         tickers: List[str],
                         return_horizon: str = '30m') -> Dict:
        """
        Train all enabled methods.

        Args:
            tweets_df: DataFrame with tweet data
            returns_df: DataFrame with forward returns
            tickers: List of ticker symbols
            return_horizon: Return horizon

        Returns:
            Dictionary with training results from each method
        """
        print(f"\n{'='*80}")
        print("METHOD 5: Ensemble Training (Training all methods)")
        print(f"{'='*80}")

        results = {}

        # Train correlation method
        if self.use_correlation:
            print("\nTraining correlation method...")
            try:
                corr_results = self.correlation_scorer.discover_relationships(
                    tweets_df, returns_df, tickers, return_horizon=return_horizon
                )
                results['correlation'] = corr_results
            except Exception as e:
                print(f"  Error training correlation: {e}")
                self.use_correlation = False

        # Train bag-of-words method
        if self.use_bow:
            print("\nTraining bag-of-words method...")
            try:
                bow_results = self.bow_scorer.discover_keywords(
                    tweets_df, returns_df, tickers, return_horizon=return_horizon
                )
                results['bag_of_words'] = bow_results
            except Exception as e:
                print(f"  Error training bag-of-words: {e}")
                self.use_bow = False

        # Train embedding method
        if self.use_embedding:
            print("\nTraining embedding method...")
            try:
                emb_results = self.embedding_scorer.train_models(
                    tweets_df, returns_df, tickers, return_horizon=return_horizon
                )
                results['embedding'] = emb_results
            except Exception as e:
                print(f"  Error training embedding: {e}")
                self.use_embedding = False

        # LLM doesn't need training (zero-shot)
        if self.use_llm:
            print("\nLLM method ready (zero-shot, no training needed)")

        return results

    def score_tweet_ticker(self,
                          tweet_row: pd.Series,
                          ticker: str,
                          returns_df: Optional[pd.DataFrame] = None,
                          return_horizon: str = '30m') -> Dict:
        """
        Score tweet-ticker pair using ensemble of methods.

        Args:
            tweet_row: Single row from tweets DataFrame
            ticker: Ticker symbol
            returns_df: Optional returns DataFrame (needed for some methods)
            return_horizon: Return horizon

        Returns:
            Dictionary with ensemble score and individual method scores
        """
        method_scores = {}

        # Get scores from each method
        if self.use_correlation:
            try:
                corr_score = self.correlation_scorer.score_tweet_ticker(
                    tweet_row, ticker, return_horizon
                )
                method_scores['correlation'] = corr_score
            except:
                pass

        if self.use_bow:
            try:
                tweet_text = tweet_row.get('tweet_content', '')
                # Handle duplicate columns from merge (value may be a Series)
                if isinstance(tweet_text, pd.Series):
                    tweet_text = tweet_text.iloc[0]
                if pd.notna(tweet_text):
                    bow_score = self.bow_scorer.score_tweet_ticker(str(tweet_text), ticker)
                    method_scores['bag_of_words'] = bow_score
            except:
                pass

        if self.use_embedding:
            try:
                emb_score = self.embedding_scorer.score_tweet_ticker(
                    tweet_row, ticker, returns_df, return_horizon, use_similarity=False
                )
                method_scores['embedding'] = emb_score
            except:
                pass

        if self.use_llm:
            try:
                tweet_text = tweet_row.get('tweet_content', '')
                # Handle duplicate columns from merge (value may be a Series)
                if isinstance(tweet_text, pd.Series):
                    tweet_text = tweet_text.iloc[0]
                if pd.notna(tweet_text):
                    llm_score = self.llm_scorer.score_tweet_ticker(str(tweet_text), ticker)
                    method_scores['llm'] = llm_score
            except:
                pass

        # Use stacking if available and enabled
        if self.use_stacking and self.meta_model is not None:
            # Prepare features for meta-model
            method_features = []
            for method_name in ['correlation', 'bag_of_words', 'embedding', 'llm']:
                if method_name in method_scores:
                    score = method_scores[method_name]
                    method_features.extend([
                        score.get('influence_score', 0.0),
                        score.get('confidence', 0.0),
                        score.get('volatility_score', 0.0)
                    ])
                else:
                    method_features.extend([0.0, 0.0, 0.0])
            
            # Predict with meta-model
            try:
                import numpy as np
                features_array = np.array(method_features).reshape(1, -1)
                influence_score = float(self.meta_model.predict(features_array)[0])
            except:
                # Fallback to weighted average
                influence_score = self._weighted_average(method_scores)
        else:
            # Use dynamic or static weighted average (fallback to static if dynamic fails)
            try:
                influence_score = self._weighted_average(method_scores, use_dynamic=True)
            except (AttributeError, KeyError):
                influence_score = self._weighted_average(method_scores, use_dynamic=False)
        
        # Calculate other metrics
        volatility_score = 0.0
        confidence_sum = 0.0
        total_weight = 0.0
        
        # Get dynamic weights if enabled
        current_weights = self._get_dynamic_weights() if len(method_scores) > 0 else self.weights
        
        for method_name, score in method_scores.items():
            weight = current_weights.get(method_name, 0.0)
            if weight > 0:
                volatility_score += score.get('volatility_score', 0.0) * weight
                confidence_sum += score.get('confidence', 0.0) * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            volatility_score = volatility_score / total_weight
            confidence = confidence_sum / total_weight if len(method_scores) > 0 else 0.0
        else:
            confidence = 0.0

        return {
            'influence_score': float(influence_score),
            'volatility_score': float(volatility_score),
            'confidence': float(confidence),
            'method': 'ensemble',
            'method_scores': method_scores,
            'n_methods': len(method_scores),
            'weights_used': current_weights
        }
    
    def _weighted_average(self, method_scores: Dict, use_dynamic: bool = False) -> float:
        """Calculate weighted average of method scores."""
        influence_score = 0.0
        total_weight = 0.0
        
        # Get weights (dynamic or static)
        if use_dynamic:
            weights = self._get_dynamic_weights()
        else:
            weights = self.weights
        
        for method_name, score in method_scores.items():
            weight = weights.get(method_name, 0.0)
            if weight > 0 and 'influence_score' in score:
                influence_score += score['influence_score'] * weight
                total_weight += weight
        
        if total_weight > 0:
            return influence_score / total_weight
        else:
            return 0.0
    
    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance."""
        if not any(self.method_performance.values()):
            return self.weights  # Fallback to static weights
        
        # Calculate average performance per method
        method_avg_perf = {}
        for method_name, perf_list in self.method_performance.items():
            if len(perf_list) > 0:
                # Use absolute correlation/performance
                method_avg_perf[method_name] = np.mean([abs(p) for p in perf_list[-self.performance_window:]])
            else:
                method_avg_perf[method_name] = 0.0
        
        # Normalize to weights
        total_perf = sum(method_avg_perf.values())
        if total_perf > 0:
            dynamic_weights = {k: v / total_perf for k, v in method_avg_perf.items()}
            # Blend with static weights (70% dynamic, 30% static)
            blended_weights = {}
            for method_name in self.weights.keys():
                dynamic_w = dynamic_weights.get(method_name, 0.0)
                static_w = self.weights.get(method_name, 0.0)
                blended_weights[method_name] = 0.7 * dynamic_w + 0.3 * static_w
            
            # Renormalize
            total = sum(blended_weights.values())
            if total > 0:
                return {k: v / total for k, v in blended_weights.items()}
        
        return self.weights
    
    def update_performance(self, method_name: str, actual_return: float, predicted_score: float):
        """
        Update performance tracking for a method.
        
        Args:
            method_name: Name of the method
            actual_return: Actual return achieved
            predicted_score: Predicted influence score
        """
        if method_name in self.method_performance:
            # Calculate correlation/performance metric
            # Use absolute correlation between prediction and actual return
            if abs(predicted_score) > 0 and abs(actual_return) > 0:
                # Simple performance metric: how well did prediction match direction?
                direction_match = 1.0 if (predicted_score * actual_return) > 0 else -1.0
                magnitude_match = 1.0 - abs(abs(predicted_score) - abs(actual_return)) / max(abs(predicted_score), abs(actual_return), 0.001)
                performance = direction_match * magnitude_match
            else:
                performance = 0.0
            
            self.method_performance[method_name].append(performance)
            
            # Keep only recent performance
            if len(self.method_performance[method_name]) > self.performance_window:
                self.method_performance[method_name] = self.method_performance[method_name][-self.performance_window:]
    
    def enable_stacking(self, tweets_df: pd.DataFrame, returns_df: pd.DataFrame, 
                       tickers: List[str], return_horizon: str = '30m'):
        """
        Enable stacking by training a meta-model.
        
        Args:
            tweets_df: Training tweets
            returns_df: Forward returns
            tickers: List of tickers
            return_horizon: Return horizon
        """
        print("\nTraining stacking meta-model...")
        
        try:
            # Try to use XGBoost if available, otherwise use linear regression
            try:
                from xgboost import XGBRegressor
                self.meta_model = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                use_xgb = True
            except ImportError:
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0)
                use_xgb = False
            
            # Collect training data
            X_train = []
            y_train = []
            
            sample_size = min(1000, len(tweets_df))  # Limit for speed
            sample_tweets = tweets_df.sample(n=sample_size, random_state=42)
            
            for _, row in sample_tweets.iterrows():
                ticker = row.get('ticker', tickers[0] if tickers else 'SPY')
                
                # Get predictions from all methods
                method_features = []
                for method_name in ['correlation', 'bag_of_words', 'embedding', 'llm']:
                    try:
                        if method_name == 'correlation' and self.use_correlation:
                            score = self.correlation_scorer.score_tweet_ticker(row, ticker, return_horizon)
                        elif method_name == 'bag_of_words' and self.use_bow:
                            tweet_text = row.get('tweet_content', '')
                            score = self.bow_scorer.score_tweet_ticker(tweet_text, ticker) if pd.notna(tweet_text) else {'influence_score': 0.0, 'confidence': 0.0}
                        elif method_name == 'embedding' and self.use_embedding:
                            score = self.embedding_scorer.score_tweet_ticker(row, ticker, returns_df, return_horizon, use_similarity=False)
                        elif method_name == 'llm' and self.use_llm:
                            tweet_text = row.get('tweet_content', '')
                            score = self.llm_scorer.score_tweet_ticker(tweet_text, ticker) if pd.notna(tweet_text) else {'influence_score': 0.0, 'confidence': 0.0}
                        else:
                            score = {'influence_score': 0.0, 'confidence': 0.0, 'volatility_score': 0.0}
                        
                        method_features.extend([
                            score.get('influence_score', 0.0),
                            score.get('confidence', 0.0),
                            score.get('volatility_score', 0.0)
                        ])
                    except:
                        method_features.extend([0.0, 0.0, 0.0])
                
                # Get actual return
                ret_col = f'{ticker}_{return_horizon}'
                if ret_col in returns_df.columns and 'tweet_id' in returns_df.columns:
                    tweet_id = row.get('tweet_id')
                    if tweet_id is not None:
                        actual_returns = returns_df[returns_df['tweet_id'] == tweet_id][ret_col]
                        if len(actual_returns) > 0:
                            actual_return = actual_returns.iloc[0]
                            X_train.append(method_features)
                            y_train.append(actual_return)
            
            if len(X_train) > 10:
                import numpy as np
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                self.meta_model.fit(X_train, y_train)
                self.use_stacking = True
                print(f"  ✓ Trained stacking meta-model ({'XGBoost' if use_xgb else 'Ridge'} regression) on {len(X_train)} samples")
            else:
                print("  Warning: Not enough training data for stacking")
                self.use_stacking = False
        except Exception as e:
            print(f"  Warning: Stacking training failed: {e}")
            self.use_stacking = False

    def optimize_weights(self,
                       tweets_df: pd.DataFrame,
                       returns_df: pd.DataFrame,
                       tickers: List[str],
                       return_horizon: str = '30m',
                       validation_split: float = 0.2) -> Dict:
        """
        Optimize ensemble weights based on validation performance.

        Args:
            tweets_df: DataFrame with tweet data
            returns_df: DataFrame with forward returns
            tickers: List of ticker symbols
            return_horizon: Return horizon
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with optimized weights
        """
        print("\nOptimizing ensemble weights...")

        # Split data
        n_samples = len(tweets_df)
        val_size = int(n_samples * validation_split)
        val_idx = np.random.choice(n_samples, val_size, replace=False)
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

        train_tweets = tweets_df.iloc[train_idx]
        val_tweets = tweets_df.iloc[val_idx]

        # Score validation set with each method
        method_predictions = {}

        for ticker in tickers:
            ret_col = f'{ticker}_{return_horizon}'
            if ret_col not in returns_df.columns:
                continue

            val_returns = returns_df[returns_df['tweet_id'].isin(val_tweets['tweet_id'])][ret_col].values

            for method_name in ['correlation', 'bag_of_words', 'embedding']:
                if method_name not in method_predictions:
                    method_predictions[method_name] = []

                # Get predictions for this method
                predictions = []
                for idx, row in val_tweets.iterrows():
                    try:
                        if method_name == 'correlation':
                            score = self.correlation_scorer.score_tweet_ticker(row, ticker, return_horizon)
                        elif method_name == 'bag_of_words':
                            score = self.bow_scorer.score_tweet_ticker(row.get('tweet_content', ''), ticker)
                        elif method_name == 'embedding':
                            score = self.embedding_scorer.score_tweet_ticker(row, ticker)
                        else:
                            continue

                        predictions.append(score['influence_score'])
                    except:
                        predictions.append(0.0)

                if len(predictions) == len(val_returns):
                    # Calculate correlation
                    corr = np.corrcoef(predictions, val_returns)[0, 1] if len(predictions) > 1 else 0.0
                    method_predictions[method_name].append(abs(corr))

        # Calculate average correlation per method
        method_correlations = {}
        for method_name, corrs in method_predictions.items():
            if len(corrs) > 0:
                method_correlations[method_name] = np.mean(corrs)

        # Set weights proportional to correlations
        total_corr = sum(method_correlations.values())
        if total_corr > 0:
            self.weights = {k: v / total_corr for k, v in method_correlations.items()}
        else:
            # Fallback to equal weights
            n_methods = len(method_correlations)
            self.weights = {k: 1.0 / n_methods for k in method_correlations.keys()}

        print(f"  Optimized weights: {self.weights}")

        return self.weights
