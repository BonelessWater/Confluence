"""
Robust tweet-ticker relevance filtering.

Checks whether a tweet is genuinely related to a specific ticker before
using it as a trading signal. Prevents noise from irrelevant tweets
(e.g., election updates, personal endorsements) being used as signals.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Per-ticker relevance specification
# ---------------------------------------------------------------------------
# Each entry has:
#   required  – at least one term must match (otherwise score=0)
#   primary   – high-weight terms (more specific to the asset)
#   secondary – lower-weight supporting terms
#   global_ok – if True, very broad macro tweets score partial credit
# ---------------------------------------------------------------------------

TICKER_SPEC: Dict[str, Dict] = {
    # ── Semiconductors / Tech ────────────────────────────────────────────
    'AMD': {
        'required': [
            'semiconductor', 'chip', 'chips', 'gpu', 'cpu', 'processor',
            'amd', 'advanced micro', 'ai chip', 'data center', 'foundry',
            'tsmc', 'export control', 'chip ban', 'tech stock', 'nvidia',
            'silicon', 'fabricat',
        ],
        'primary': [
            'amd', 'advanced micro devices', 'gpu', 'cpu', 'processor',
            'semiconductor', 'chip', 'chips', 'chip ban', 'chip restriction',
            'export control', 'data center', 'ai chip', 'tsmc', 'foundry',
            'silicon', 'fabricat', 'intel', 'arm ', 'microchip',
        ],
        'secondary': [
            'artificial intelligence', 'ai ', 'technology', 'tech', 'china tech',
            'huawei', 'supply chain', 'manufacturing', 'pc market',
        ],
        'global_ok': False,
    },
    'NVDA': {
        'required': [
            'nvidia', 'gpu', 'ai chip', 'semiconductor', 'chip', 'chips',
            'data center', 'artificial intelligence', 'h100', 'a100',
            'blackwell', 'cuda', 'jensen', 'export control', 'chip ban',
            'deep learning', 'machine learning', 'foundry', 'tsmc',
        ],
        'primary': [
            'nvidia', 'nvda', 'h100', 'a100', 'blackwell', 'jensen huang',
            'cuda', 'gpu', 'ai chip', 'chip ban', 'chip restriction',
            'export control', 'semiconductor', 'data center',
        ],
        'secondary': [
            'artificial intelligence', 'ai ', 'deep learning', 'machine learning',
            'cloud computing', 'technology', 'tech', 'huawei', 'tsmc',
            'chip', 'chips', 'china tech',
        ],
        'global_ok': False,
    },
    'QCOM': {
        'required': [
            'qualcomm', '5g', 'wireless', 'smartphone chip', 'modem',
            'chip', 'chips', 'semiconductor', 'arm ', 'huawei',
            'handset', 'telecom', 'patent', 'mobile chip', 'snapdragon',
            'export control', 'china ban',
        ],
        'primary': [
            'qualcomm', 'qcom', 'snapdragon', '5g', 'wireless', 'modem',
            'smartphone chip', 'handset', 'patent', 'telecom', 'arm license',
        ],
        'secondary': [
            'chip', 'chips', 'semiconductor', 'huawei', 'china ban',
            'mobile', 'smartphone', 'technology', 'tech', 'export control',
        ],
        'global_ok': False,
    },

    # ── Broad US Equity ──────────────────────────────────────────────────
    'SPY': {
        'required': [
            'stock market', 'wall street', 's&p', 'market crash', 'bull market',
            'bear market', 'equity', 'stocks', 'dow', 'nasdaq', 'economy',
            'gdp', 'recession', 'growth', 'tariff', 'trade war', 'trade deal',
            'job', 'unemployment', 'earnings', 'corporate tax', 'fiscal',
            'interest rate', 'federal reserve', 'fed ', 'inflation',
        ],
        'primary': [
            'stock market', 'wall street', 's&p 500', 'market crash', 'bull market',
            'bear market', 'stocks', 'equities', 'nasdaq', 'dow jones',
            'market rally', 'market decline',
        ],
        'secondary': [
            'economy', 'gdp', 'growth', 'recession', 'tariff', 'trade war',
            'trade deal', 'jobs', 'unemployment', 'earnings', 'inflation',
            'federal reserve', 'interest rate', 'corporate tax', 'fiscal',
        ],
        'global_ok': True,
    },
    'DIA': {
        'required': [
            'dow', 'dow jones', 'industrial', 'blue chip', 'manufacturing',
            'factory', 'jobs', 'unemployment', 'earnings', 'stock market',
            'market', 'tariff', 'trade', 'economy', 'gdp', 'recession',
        ],
        'primary': [
            'dow jones', 'dow 30', 'industrial average', 'blue chip',
            'manufacturing', 'factory orders', 'industrial production',
        ],
        'secondary': [
            'jobs', 'unemployment', 'earnings', 'stock market', 'economy',
            'tariff', 'trade', 'gdp', 'recession',
        ],
        'global_ok': True,
    },

    # ── Bonds / Rates ────────────────────────────────────────────────────
    'TLT': {
        'required': [
            'federal reserve', 'fed ', 'interest rate', 'treasury', 'bond',
            'yield', 'rate hike', 'rate cut', 'inflation', 'cpi',
            'monetary policy', 'quantitative easing', 'jerome powell',
            'debt', 'deficit', 'long-term rate', '10-year', '30-year',
        ],
        'primary': [
            'federal reserve', 'fed rate', 'interest rate', 'rate hike',
            'rate cut', 'treasury bond', 'long-term bond', '10-year yield',
            '30-year', 'bond yield', 'jerome powell', 'quantitative easing',
            'monetary policy',
        ],
        'secondary': [
            'inflation', 'cpi', 'deficit', 'debt', 'treasury', 'yield',
            'bond market', 'recession', 'economy', 'gdp',
        ],
        'global_ok': False,
    },
    'IEF': {
        'required': [
            'federal reserve', 'fed ', 'interest rate', 'treasury', 'bond',
            'yield', 'rate hike', 'rate cut', 'inflation', 'cpi',
            'monetary policy', 'jerome powell', '7-year', '10-year',
            'medium-term', 'note',
        ],
        'primary': [
            'federal reserve', 'fed rate', 'interest rate', 'rate hike',
            'rate cut', 'treasury note', '7-year', '10-year yield',
            'bond yield', 'jerome powell', 'monetary policy',
        ],
        'secondary': [
            'inflation', 'cpi', 'deficit', 'treasury', 'yield',
            'bond market', 'recession', 'economy',
        ],
        'global_ok': False,
    },
    'SHY': {
        'required': [
            'federal reserve', 'fed ', 'interest rate', 'treasury', 'bond',
            'yield', 'rate hike', 'rate cut', 'fed funds', 'money market',
            'short-term rate', 't-bill', '2-year', '3-year', '1-year',
            'jerome powell', 'monetary policy',
        ],
        'primary': [
            'federal reserve', 'fed funds rate', 'short-term rate', 't-bill',
            '2-year yield', '3-year', 'rate hike', 'rate cut',
            'jerome powell', 'money market',
        ],
        'secondary': [
            'inflation', 'treasury', 'yield', 'interest rate', 'monetary policy',
        ],
        'global_ok': False,
    },

    # ── Commodities ──────────────────────────────────────────────────────
    'GLD': {
        'required': [
            'gold price', 'gold reserve', 'gold market', 'buy gold', 'gold etf',
            'gold bullion', 'buy bullion', 'bullion', 'precious metal',
            'safe haven', 'inflation hedge', 'gold hits', 'gold surges',
            'gold falls', 'gold rises', 'gold drops', 'gold rally',
            'gold mining', 'gold miner',
        ],
        'primary': [
            'gold price', 'gold reserve', 'bullion', 'precious metal',
            'inflation hedge', 'safe haven', 'gold bullion', 'gold mining',
        ],
        'secondary': [
            'dollar', 'inflation', 'war', 'conflict', 'geopolitical',
            'crisis', 'uncertainty', 'currency', 'recession',
        ],
        'global_ok': False,
    },
    'USO': {
        'required': [
            'oil', 'crude', 'petroleum', 'opec', 'barrel', 'brent', 'wti',
            'gasoline', 'gas price', 'pipeline', 'lng', 'energy',
            'russia oil', 'saudi', 'iran', 'oil production', 'oil price',
            'energy independence', 'drill', 'shale',
        ],
        'primary': [
            'oil', 'crude oil', 'oil price', 'opec', 'barrel', 'brent',
            'wti', 'petroleum', 'gasoline', 'gas price', 'pipeline',
            'lng', 'oil production', 'energy', 'drill', 'shale',
        ],
        'secondary': [
            'russia', 'saudi arabia', 'iran', 'energy independence',
            'inflation', 'supply chain', 'refinery',
        ],
        'global_ok': False,
    },

    # ── FX ───────────────────────────────────────────────────────────────
    'UUP': {
        'required': [
            'dollar', 'usd', 'dxy', 'dollar index', 'strong dollar',
            'weak dollar', 'dollar value', 'currency', 'exchange rate',
            'forex', 'federal reserve', 'fed rate', 'trade deficit',
            'trade surplus',
        ],
        'primary': [
            'dollar', 'usd', 'dxy', 'dollar index', 'strong dollar',
            'weak dollar', 'dollar value', 'exchange rate', 'forex',
            'trade deficit', 'trade surplus',
        ],
        'secondary': [
            'federal reserve', 'fed rate', 'interest rate', 'inflation',
            'currency', 'trade war', 'tariff', 'economy',
        ],
        'global_ok': False,
    },

    # ── EM / Sector ETFs ─────────────────────────────────────────────────
    'EWW': {
        'required': [
            'mexico', 'mexican', 'nafta', 'usmca', 'border', 'nearshoring',
            'peso', 'latin america', 'tariff mexico', 'mexico tariff',
            'immigration', 'deportation', 'wall', 'trade mexico',
            'amlo', 'claudia sheinbaum', 'monterrey', 'tijuana',
        ],
        'primary': [
            'mexico', 'mexican', 'usmca', 'nafta', 'peso', 'nearshoring',
            'amlo', 'claudia sheinbaum', 'border wall', 'tariff mexico',
            'mexico tariff', 'deportation mexico',
        ],
        'secondary': [
            'border', 'immigration', 'deportation', 'latin america',
            'trade', 'tariff', 'manufacturing',
        ],
        'global_ok': False,
    },
    'CYB': {
        'required': [
            'china', 'chinese', 'yuan', 'renminbi', 'rmb', 'cny', 'pboc',
            'people\'s bank', 'trade war', 'tariff china', 'china tariff',
            'hong kong', 'shanghai', 'beijing', 'xi jinping',
            'china currency', 'devalue', 'export china',
        ],
        'primary': [
            'china', 'chinese', 'yuan', 'renminbi', 'rmb', 'cny', 'pboc',
            'xi jinping', 'trade war', 'tariff china', 'china tariff',
            'hong kong', 'devalue yuan', 'currency manipulation',
        ],
        'secondary': [
            'tariff', 'trade', 'export', 'import', 'supply chain',
            'huawei', 'chip ban', 'china ban',
        ],
        'global_ok': False,
    },
}

# Macro topics that affect multiple tickers when mentioned
MACRO_TOPICS = {
    'tariff': ['SPY', 'DIA', 'EWW', 'CYB', 'USO', 'UUP'],
    'trade war': ['SPY', 'DIA', 'EWW', 'CYB', 'USO', 'UUP'],
    'trade deal': ['SPY', 'DIA', 'EWW', 'CYB'],
    'federal reserve': ['TLT', 'IEF', 'SHY', 'UUP', 'GLD', 'SPY'],
    'inflation': ['TLT', 'IEF', 'GLD', 'USO', 'UUP', 'SPY'],
    'recession': ['SPY', 'DIA', 'TLT', 'GLD', 'USO'],
    'interest rate': ['TLT', 'IEF', 'SHY', 'UUP', 'GLD', 'SPY'],
    'sanctions': ['USO', 'GLD', 'CYB', 'UUP'],
    'oil': ['USO', 'UUP', 'SPY'],
    'china': ['CYB', 'AMD', 'NVDA', 'QCOM', 'SPY', 'UUP'],
    'dollar': ['UUP', 'GLD', 'USO'],
}

# Hard-exclude patterns — tweets dominated by these topics are NOT financial signals
HARD_EXCLUDE_PATTERNS = [
    r'\bvote\b', r'\belection\b', r'\bpolling\b', r'\bvoting location\b',
    r'\bprimary\b', r'\bcampaign\b', r'\bballot\b', r'\bcongratulations?\b',
    r'\bbirthday\b', r'\bmerry christmas\b', r'\bhappy new year\b',
    r'\bgod bless\b', r'\bprayer\b', r'\bpraying\b',
    r'\bsports?\b', r'\bfootball\b', r'\bbaseball\b', r'\bnfl\b',
    r'\bsuper bowl\b', r'\bchampion\b',
]

# Minimum tweet length for it to even be considered
MIN_TWEET_LENGTH = 30


class TickerRelevanceFilter:
    """
    Filters tweets to only those genuinely relevant to each ticker.

    The filter uses a keyword-matching approach with required terms (gate),
    primary terms (high weight), and secondary terms (lower weight).
    Tweets that don't mention any required terms score zero for that ticker.
    """

    def __init__(self,
                 min_relevance_score: float = 0.15,
                 require_primary_match: bool = False):
        """
        Args:
            min_relevance_score: Minimum relevance score [0, 1] to keep a
                tweet-ticker pair (0 = include all, 1 = only exact matches).
            require_primary_match: If True, a primary keyword must match
                (stricter). If False, any required keyword suffices.
        """
        self.min_relevance_score = min_relevance_score
        self.require_primary_match = require_primary_match
        self._compiled_exclude = [
            re.compile(p, re.IGNORECASE) for p in HARD_EXCLUDE_PATTERNS
        ]

    def _normalize(self, text: str) -> str:
        """Lowercase and collapse whitespace."""
        return ' '.join(text.lower().split())

    def _is_hard_excluded(self, text: str) -> bool:
        """Return True if tweet is dominated by non-financial content."""
        matches = sum(bool(p.search(text)) for p in self._compiled_exclude)
        # Hard-exclude only if multiple non-financial signals with no financial keyword
        if matches >= 2:
            has_any_financial = any(
                kw in text for kw in [
                    'market', 'stock', 'economy', 'trade', 'tariff',
                    'oil', 'gold', 'dollar', 'rate', 'inflation', 'gdp',
                ]
            )
            if not has_any_financial:
                return True
        return False

    def score_relevance(self, tweet_text: str, ticker: str) -> float:
        """
        Score how relevant a tweet is to a specific ticker.

        Returns:
            float in [0, 1]; 0 means irrelevant, 1 means highly relevant.
        """
        if not isinstance(tweet_text, str) or len(tweet_text) < MIN_TWEET_LENGTH:
            return 0.0

        text = self._normalize(tweet_text)

        if self._is_hard_excluded(text):
            return 0.0

        spec = TICKER_SPEC.get(ticker)
        if spec is None:
            return 0.0

        required = spec.get('required', [])
        primary = spec.get('primary', [])
        secondary = spec.get('secondary', [])

        # Gate: must match at least one required keyword
        has_required = any(kw in text for kw in required)
        if not has_required:
            return 0.0

        # Primary matches (weight 2)
        n_primary = sum(1 for kw in primary if kw in text)
        # Secondary matches (weight 1)
        n_secondary = sum(1 for kw in secondary if kw in text)

        if self.require_primary_match and n_primary == 0:
            return 0.0

        # Score = weighted match density (capped at 1)
        max_possible = max(1, len(primary) * 2 + len(secondary))
        raw = n_primary * 2 + n_secondary
        score = min(1.0, raw / max(1, min(max_possible, 10)))

        # Boost for direct ticker mention in tweet text
        if ticker.lower() in text:
            score = min(1.0, score + 0.3)

        return score

    def is_relevant(self, tweet_text: str, ticker: str) -> bool:
        """Return True if the tweet meets the minimum relevance threshold."""
        return self.score_relevance(tweet_text, ticker) >= self.min_relevance_score

    def filter_relevant_tweets(
        self,
        tweets_df: pd.DataFrame,
        ticker: str,
        text_col: str = 'tweet_content',
    ) -> pd.DataFrame:
        """
        Filter a DataFrame of tweets to only those relevant to `ticker`.

        Args:
            tweets_df: DataFrame with tweet data (must have `text_col`).
            ticker: Ticker symbol.
            text_col: Column containing tweet text.

        Returns:
            Filtered DataFrame with a new 'relevance_score' column.
        """
        if text_col not in tweets_df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame")

        scores = tweets_df[text_col].apply(
            lambda t: self.score_relevance(str(t) if pd.notna(t) else '', ticker)
        )
        mask = scores >= self.min_relevance_score
        result = tweets_df[mask].copy()
        result['relevance_score'] = scores[mask]
        return result

    def filter_all_tickers(
        self,
        tweets_df: pd.DataFrame,
        text_col: str = 'tweet_content',
        ticker_col: str = 'ticker',
    ) -> pd.DataFrame:
        """
        Filter a DataFrame that has one row per (tweet, ticker), keeping
        only rows where the tweet is actually relevant to its assigned ticker.

        Args:
            tweets_df: DataFrame with columns [text_col, ticker_col].
            text_col: Column containing tweet text.
            ticker_col: Column containing ticker symbol.

        Returns:
            Filtered DataFrame with a new 'relevance_score' column.
        """
        if text_col not in tweets_df.columns:
            raise ValueError(f"Column '{text_col}' not found")
        if ticker_col not in tweets_df.columns:
            raise ValueError(f"Column '{ticker_col}' not found")

        scores = tweets_df.apply(
            lambda row: self.score_relevance(
                str(row[text_col]) if pd.notna(row[text_col]) else '',
                row[ticker_col]
            ),
            axis=1,
        )
        mask = scores >= self.min_relevance_score
        result = tweets_df[mask].copy()
        result['relevance_score'] = scores[mask]
        return result

    def report_coverage(
        self,
        tweets_df: pd.DataFrame,
        text_col: str = 'tweet_content',
        ticker_col: str = 'ticker',
    ) -> pd.DataFrame:
        """
        Generate a summary of how many tweets pass the relevance filter
        per ticker.

        Returns:
            DataFrame with columns: ticker, total_tweets, relevant_tweets,
            pct_relevant, avg_score.
        """
        rows = []
        for ticker in tweets_df[ticker_col].unique():
            subset = tweets_df[tweets_df[ticker_col] == ticker]
            scores = subset[text_col].apply(
                lambda t: self.score_relevance(
                    str(t) if pd.notna(t) else '', ticker
                )
            )
            n_relevant = (scores >= self.min_relevance_score).sum()
            rows.append({
                'ticker': ticker,
                'total_tweets': len(subset),
                'relevant_tweets': int(n_relevant),
                'pct_relevant': round(n_relevant / max(1, len(subset)) * 100, 2),
                'avg_score': round(scores.mean(), 4),
                'avg_score_relevant': round(
                    scores[scores >= self.min_relevance_score].mean()
                    if n_relevant > 0 else 0.0,
                    4,
                ),
            })
        return pd.DataFrame(rows).sort_values('pct_relevant', ascending=False)

    def get_top_relevant_tweets(
        self,
        tweets_df: pd.DataFrame,
        ticker: str,
        n: int = 20,
        text_col: str = 'tweet_content',
    ) -> pd.DataFrame:
        """Return the N most relevant tweets for a given ticker."""
        if text_col not in tweets_df.columns:
            return pd.DataFrame()
        if 'ticker' in tweets_df.columns:
            subset = tweets_df[tweets_df['ticker'] == ticker]
        else:
            subset = tweets_df
        scores = subset[text_col].apply(
            lambda t: self.score_relevance(str(t) if pd.notna(t) else '', ticker)
        )
        top_idx = scores.nlargest(n).index
        result = subset.loc[top_idx].copy()
        result['relevance_score'] = scores[top_idx]
        return result.sort_values('relevance_score', ascending=False)
