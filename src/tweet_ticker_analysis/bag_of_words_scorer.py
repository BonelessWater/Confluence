"""
Option 2: Bag-of-Words with Ticker-Specific Scoring

Extracts keywords/n-grams and scores their relationship to each ticker.
Refined version with statistical filtering, directional scoring, and TF-IDF weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')


class BagOfWordsScorer:
    """
    Score tweet-ticker relationships using bag-of-words approach.
    
    Refinements over baseline:
    - Statistical significance filtering (p-value gating)
    - Top-N keyword selection per ticker (reduces noise)
    - Trigram support for multi-word phrases
    - T-stat weighted scoring (emphasizes strong signals)
    - Directional keyword separation (bullish vs bearish)
    - Higher min_occurrences for robustness
    """

    def __init__(self,
                 max_features: int = 2000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 max_keywords_per_ticker: int = 50,
                 p_value_threshold: float = 0.10,
                 min_occurrences: int = 10,
                 train_ratio: float = 0.7):
        """
        Initialize bag-of-words scorer.

        Args:
            max_features: Maximum TF-IDF features to extract
            ngram_range: Range of n-grams (min, max) — (1,3) captures unigrams through trigrams
            max_keywords_per_ticker: Keep only top N keywords per ticker by significance
            p_value_threshold: Only keep keywords with p-value below this
            min_occurrences: Minimum times a keyword must appear to be considered
            train_ratio: Fraction of data (by time) for training. Prevents look-ahead bias.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_keywords_per_ticker = max_keywords_per_ticker
        self.p_value_threshold = p_value_threshold
        self.min_occurrences = min_occurrences
        self.train_ratio = train_ratio
        self.train_end_time = None
        self.vectorizer = None
        self.keyword_scores = {}  # {(keyword, ticker): scores}
        self.vocabulary = None
        # Per-ticker top keywords for fast scoring
        self._ticker_keywords = {}  # {ticker: [(keyword, scores), ...]}

    def discover_keywords(self,
                         tweets_df: pd.DataFrame,
                         returns_df: pd.DataFrame,
                         tickers: List[str],
                         return_horizon: str = '30m',
                         min_occurrences: int = None) -> Dict:
        """
        Discover keyword-ticker relationships with statistical filtering.

        Args:
            tweets_df: DataFrame with tweet content
            returns_df: DataFrame with forward returns
            tickers: List of ticker symbols
            return_horizon: Return horizon to analyze
            min_occurrences: Override minimum occurrences (uses self.min_occurrences if None)

        Returns:
            Dictionary with keyword-ticker scores
        """
        if min_occurrences is None:
            min_occurrences = self.min_occurrences

        print(f"\n{'='*80}")
        print("METHOD 2: Bag-of-Words Keyword Discovery (Refined)")
        print(f"{'='*80}")

        if 'tweet_content' not in tweets_df.columns:
            raise ValueError("tweets_df must contain 'tweet_content' column")

        # LOOK-AHEAD FIX: Use only training period for keyword discovery
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

        # Clean and prepare text (use train data only)
        texts = train_tweets['tweet_content'].fillna('').astype(str).tolist()
        
        # Initialize vectorizer with trigrams
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=min_occurrences,
            max_df=0.8,  # Ignore terms in >80% of docs (too common to be informative)
            sublinear_tf=True,  # Apply log normalization to TF (reduces impact of very frequent terms)
        )

        print(f"Extracting features (max_features={self.max_features}, "
              f"ngram_range={self.ngram_range}, min_df={min_occurrences}, max_df=0.8)...")
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.vocabulary = self.vectorizer.get_feature_names_out()
        except Exception as e:
            print(f"  Warning: TF-IDF extraction failed: {e}")
            return self._simple_keyword_extraction(train_tweets, returns_df, tickers, return_horizon)

        print(f"  Extracted {len(self.vocabulary)} features")

        # Convert to DataFrame - add tweet_id and ticker for correct merge (train data only)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vocabulary,
            index=train_tweets.index
        )
        tfidf_df['tweet_id'] = train_tweets['tweet_id'].values
        tfidf_df['ticker'] = train_tweets['ticker'].values

        results = {}
        total_significant = 0

        for ticker in tickers:
            ret_col = f'{ticker}_{return_horizon}'
            
            if ret_col not in returns_df.columns:
                print(f"  Warning: {ret_col} not found, skipping {ticker}")
                continue

            # Align data: merge on tweet_id, filter to current ticker
            ticker_tfidf = tfidf_df[tfidf_df['ticker'] == ticker].drop(columns=['ticker'])
            returns_subset = returns_df[['tweet_id', ret_col]].dropna(subset=[ret_col])
            aligned_df = pd.merge(
                ticker_tfidf,
                returns_subset,
                on='tweet_id',
                how='inner'
            )

            if len(aligned_df) == 0:
                continue

            aligned_df = aligned_df.dropna(subset=[ret_col])
            
            if len(aligned_df) < 20:
                continue

            overall_mean = aligned_df[ret_col].mean()
            overall_std = aligned_df[ret_col].std()

            ticker_keyword_scores = {}

            for keyword in self.vocabulary:
                # Get tweets containing this keyword (TF-IDF > 0)
                keyword_mask = aligned_df[keyword] > 0
                keyword_returns = aligned_df.loc[keyword_mask, ret_col]
                keyword_tfidf_weights = aligned_df.loc[keyword_mask, keyword]

                if len(keyword_returns) < min_occurrences:
                    continue

                # Calculate statistics
                mean_return = keyword_returns.mean()
                std_return = keyword_returns.std()
                
                # T-test: is the mean return when keyword present different from overall mean?
                if std_return > 0 and len(keyword_returns) >= 3:
                    t_stat, p_value = stats.ttest_1samp(keyword_returns, overall_mean)
                else:
                    t_stat, p_value = 0.0, 1.0

                # ** P-value gate: skip noisy keywords **
                if p_value > self.p_value_threshold:
                    continue

                hit_rate = (keyword_returns > 0).mean()
                
                # Excess return over baseline
                excess_return = mean_return - overall_mean
                
                # Volatility impact
                vol_impact = std_return - overall_std if overall_std > 0 else 0.0

                # Mean TF-IDF weight (how strongly this keyword appears)
                mean_tfidf = keyword_tfidf_weights.mean()

                # Information coefficient: correlation between TF-IDF weight and return
                if len(keyword_returns) >= 10 and keyword_tfidf_weights.std() > 0:
                    ic, ic_pvalue = stats.spearmanr(keyword_tfidf_weights, keyword_returns)
                else:
                    ic, ic_pvalue = 0.0, 1.0

                ticker_keyword_scores[keyword] = {
                    'mean_return': mean_return,
                    'excess_return': excess_return,
                    'std_return': std_return,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'hit_rate': hit_rate,
                    'vol_impact': vol_impact,
                    'n_occurrences': len(keyword_returns),
                    'mean_tfidf': mean_tfidf,
                    'information_coefficient': ic,
                    'direction': 'bullish' if excess_return > 0 else 'bearish',
                }

                self.keyword_scores[(keyword, ticker)] = ticker_keyword_scores[keyword]

            # ** Top-N selection: keep only the most significant keywords **
            sorted_keywords = sorted(
                ticker_keyword_scores.items(),
                key=lambda x: abs(x[1]['t_stat']),
                reverse=True
            )[:self.max_keywords_per_ticker]

            # Store per-ticker top keywords for fast scoring
            self._ticker_keywords[ticker] = sorted_keywords

            n_significant = len(sorted_keywords)
            total_significant += n_significant
            n_bullish = sum(1 for _, s in sorted_keywords if s['direction'] == 'bullish')
            n_bearish = n_significant - n_bullish

            results[ticker] = {
                'top_keywords': sorted_keywords[:20],
                'all_keywords': {k: v for k, v in sorted_keywords},
                'n_samples': len(aligned_df),
                'n_significant': n_significant,
                'n_bullish': n_bullish,
                'n_bearish': n_bearish,
            }

            print(f"\n  {ticker}:")
            print(f"    Samples: {len(aligned_df)}")
            print(f"    Significant keywords (p<{self.p_value_threshold}): {n_significant} "
                  f"({n_bullish} bullish, {n_bearish} bearish)")
            print(f"    Top 5 keywords by |t-stat|:")
            for kw, scores in sorted_keywords[:5]:
                direction = "+" if scores['direction'] == 'bullish' else "-"
                print(f"      [{direction}] '{kw}': excess={scores['excess_return']:.6f}, "
                      f"t={scores['t_stat']:.2f}, p={scores['p_value']:.4f}, "
                      f"n={scores['n_occurrences']}, hit={scores['hit_rate']:.1%}")

        print(f"\n  Total significant keywords across all tickers: {total_significant}")
        return results

    def get_backtest_tweets(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """Filter tweets to backtest period only (after train_end_time). Prevents look-ahead."""
        if self.train_end_time is None or 'entry_time' not in tweets_df.columns:
            return tweets_df
        return tweets_df[tweets_df['entry_time'] >= self.train_end_time].copy()

    def _simple_keyword_extraction(self,
                                   tweets_df: pd.DataFrame,
                                   returns_df: pd.DataFrame,
                                   tickers: List[str],
                                   return_horizon: str) -> Dict:
        """Fallback simple keyword extraction."""
        print("  Using simple keyword extraction (fallback)")
        
        all_words = []
        for text in tweets_df['tweet_content'].fillna(''):
            words = re.findall(r'\b[a-z]+\b', text.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        common_words = [w for w, c in word_counts.most_common(self.max_features) if c >= self.min_occurrences]
        self.vocabulary = np.array(common_words)
        
        print(f"  Extracted {len(self.vocabulary)} keywords")
        
        results = {}
        
        for ticker in tickers:
            ret_col = f'{ticker}_{return_horizon}'
            if ret_col not in returns_df.columns:
                continue
            
            # Filter to current ticker for correct alignment
            ticker_tweets = tweets_df[tweets_df['ticker'] == ticker][['tweet_content', 'tweet_id']]
            returns_subset = returns_df[['tweet_id', ret_col]].dropna(subset=[ret_col])
            aligned_df = pd.merge(
                ticker_tweets,
                returns_subset,
                on='tweet_id',
                how='inner'
            ).dropna(subset=[ret_col])
            
            if len(aligned_df) < 20:
                continue

            overall_mean = aligned_df[ret_col].mean()
            ticker_keyword_scores = {}
            
            for keyword in self.vocabulary:
                keyword_mask = aligned_df['tweet_content'].str.contains(keyword, case=False, na=False)
                keyword_returns = aligned_df.loc[keyword_mask, ret_col]
                
                if len(keyword_returns) < self.min_occurrences:
                    continue
                
                mean_return = keyword_returns.mean()
                excess_return = mean_return - overall_mean
                if keyword_returns.std() > 0:
                    t_stat, p_value = stats.ttest_1samp(keyword_returns, overall_mean)
                else:
                    t_stat, p_value = 0.0, 1.0

                if p_value > self.p_value_threshold:
                    continue

                hit_rate = (keyword_returns > 0).mean()
                vol_impact = keyword_returns.std() - aligned_df[ret_col].std()
                
                ticker_keyword_scores[keyword] = {
                    'mean_return': mean_return,
                    'excess_return': excess_return,
                    'std_return': keyword_returns.std(),
                    't_stat': t_stat,
                    'p_value': p_value,
                    'hit_rate': hit_rate,
                    'vol_impact': vol_impact,
                    'n_occurrences': len(keyword_returns),
                    'mean_tfidf': 1.0,
                    'information_coefficient': 0.0,
                    'direction': 'bullish' if excess_return > 0 else 'bearish',
                }
                
                self.keyword_scores[(keyword, ticker)] = ticker_keyword_scores[keyword]
            
            sorted_keywords = sorted(
                ticker_keyword_scores.items(),
                key=lambda x: abs(x[1]['t_stat']),
                reverse=True
            )[:self.max_keywords_per_ticker]

            self._ticker_keywords[ticker] = sorted_keywords
            
            results[ticker] = {
                'top_keywords': sorted_keywords[:20],
                'all_keywords': {k: v for k, v in sorted_keywords},
                'n_samples': len(aligned_df),
            }
        
        return results

    def score_tweet_ticker(self,
                          tweet_text,
                          ticker: str) -> Dict:
        """
        Score a single tweet-ticker pair using discovered keywords.

        Uses t-stat-weighted scoring: keywords with stronger statistical 
        significance contribute more to the final score.

        Args:
            tweet_text: Tweet content (str) or a pd.Series row from DataFrame
            ticker: Ticker symbol

        Returns:
            Dictionary with influence score and metadata
        """
        if self.keyword_scores is None or len(self.keyword_scores) == 0:
            raise ValueError("Must call discover_keywords() first")

        if self.vocabulary is None:
            return {
                'influence_score': 0.0,
                'volatility_score': 0.0,
                'confidence': 0.0,
                'method': 'bag_of_words',
                'matched_keywords': []
            }

        # Handle pd.Series (full row from DataFrame)
        if isinstance(tweet_text, pd.Series):
            text_val = tweet_text.get('tweet_content', '')
            if isinstance(text_val, pd.Series):
                text_val = text_val.iloc[0]
            tweet_text = str(text_val) if pd.notna(text_val) else ''

        tweet_lower = tweet_text.lower() if isinstance(tweet_text, str) else ''
        
        # Use per-ticker top keywords for fast, focused scoring
        ticker_kws = self._ticker_keywords.get(ticker, [])
        
        weighted_score = 0.0
        total_weight = 0.0
        volatility_score = 0.0
        matched_keywords = []

        for keyword, scores in ticker_kws:
            if keyword in tweet_lower:
                # Weight = |t-stat|^2 — aggressively emphasizes significant keywords
                weight = scores['t_stat'] ** 2
                
                # Contribution = excess return * weight
                weighted_score += scores['excess_return'] * weight
                total_weight += weight
                volatility_score += abs(scores['vol_impact']) * abs(scores['t_stat'])
                
                matched_keywords.append({
                    'keyword': keyword,
                    'excess_return': scores['excess_return'],
                    't_stat': scores['t_stat'],
                    'p_value': scores['p_value'],
                    'direction': scores['direction'],
                    'n_occurrences': scores['n_occurrences']
                })

        # Weighted average (by t-stat^2)
        if total_weight > 0:
            influence_score = weighted_score / total_weight
            volatility_score = volatility_score / total_weight
            # Confidence based on number of significant matches and total weight
            confidence = min(1.0, len(matched_keywords) / 3.0) * min(1.0, total_weight / 10.0)
            # Require 2+ keyword matches for full signal - single matches are noisy
            if len(matched_keywords) < 2:
                influence_score *= 0.5  # Dampen weak single-keyword signals
        else:
            influence_score = 0.0
            volatility_score = 0.0
            confidence = 0.0

        return {
            'influence_score': float(influence_score),
            'volatility_score': float(volatility_score),
            'confidence': float(confidence),
            'method': 'bag_of_words',
            'matched_keywords': matched_keywords
        }
