"""
Backtesting framework for tweet-ticker relationship strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from config.settings import INITIAL_CAPITAL, HOLDING_PERIOD_MINUTES, COMMISSION_BPS, SLIPPAGE_BPS
from src.backtesting.return_calculator import RealReturnCalculator
from src.backtesting.transaction_costs import TransactionCostCalculator
from .market_regime import MarketRegimeDetector
from .position_sizing import AdaptivePositionSizer


def _build_reason(scorer, score_dict: dict, ticker: str) -> str:
    """
    Extract a human-readable reason from a scorer's score result.
    """
    if not isinstance(score_dict, dict):
        return type(scorer).__name__
    matched = score_dict.get('matched_features', [])
    if matched:
        top = matched[0]
        feat = top.get('feature', '') if isinstance(top, dict) else str(top)
        corr = top.get('correlation', 0.0) if isinstance(top, dict) else 0.0
        direction = '+' if corr >= 0 else '-'
        return f"{type(scorer).__name__}:{direction}{feat[:30]}"
    keywords = score_dict.get('matched_keywords', [])
    if keywords:
        return f"{type(scorer).__name__}:{'+'.join(str(k) for k in keywords[:2])}"
    method = score_dict.get('method', type(scorer).__name__)
    return method


class TweetStrategyBacktester:
    """
    Backtester for tweet-ticker relationship strategies.
    """

    def __init__(self,
                 initial_capital: float = INITIAL_CAPITAL,
                 holding_period_minutes: int = HOLDING_PERIOD_MINUTES,
                 apply_transaction_costs: bool = True,
                 max_trades_per_day: int = 5,
                 use_time_filter: bool = True,
                 min_hour: int = 12,
                 max_hour: int = 14,
                 use_regime_detection: bool = True,
                 use_advanced_sizing: bool = True):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            holding_period_minutes: Holding period in minutes
            apply_transaction_costs: Whether to apply transaction costs
            max_trades_per_day: Maximum number of trades per day
            use_time_filter: Whether to filter by time of day
            min_hour: Minimum hour (ET) for trading
            max_hour: Maximum hour (ET) for trading
        """
        self.initial_capital = initial_capital
        self.holding_period_minutes = holding_period_minutes
        self.apply_transaction_costs = apply_transaction_costs
        self.max_trades_per_day = max_trades_per_day
        self.use_time_filter = use_time_filter
        self.min_hour = min_hour
        self.max_hour = max_hour
        self.use_regime_detection = use_regime_detection
        self.use_advanced_sizing = use_advanced_sizing
        
        self.return_calculator = RealReturnCalculator()
        
        if apply_transaction_costs:
            self.cost_calculator = TransactionCostCalculator(
                commission_bps=COMMISSION_BPS,
                slippage_bps=SLIPPAGE_BPS
            )
        
        # Initialize regime detector if enabled
        if use_regime_detection:
            self.regime_detector = MarketRegimeDetector()
            print("  Market regime detection enabled")
        else:
            self.regime_detector = None
        
        # Initialize position sizer if enabled
        if use_advanced_sizing:
            self.position_sizer = AdaptivePositionSizer()
            print("  Advanced position sizing enabled")
        else:
            self.position_sizer = None

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax of array."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def backtest_strategy(self,
                         tweets_df: pd.DataFrame,
                         scorer,
                         tickers: List[str],
                         price_data: Dict[str, pd.DataFrame],
                         return_horizon: str = '30m',
                         min_score_threshold: float = 0.0,
                         use_multi_timeframe: bool = False,
                         returns_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Backtest a tweet-ticker scoring strategy.

        Args:
            tweets_df: DataFrame with tweet data
            scorer: Scorer instance (CorrelationDiscovery, BagOfWordsScorer, etc.)
            tickers: List of ticker symbols
            price_data: Dictionary of {ticker: price_df} with OHLCV data
            return_horizon: Return horizon for scoring
            min_score_threshold: Minimum influence score to take a trade

        Returns:
            trades_df, equity_df, final_capital
        """
        print(f"\n{'='*80}")
        print("BACKTESTING TWEET-TICKER STRATEGY")
        print(f"{'='*80}")
        print(f"Tickers: {tickers}")
        print(f"Holding period: {self.holding_period_minutes} minutes")
        print(f"Min score threshold: {min_score_threshold}")

        # Initialize
        capital = self.initial_capital
        equity_curve = []
        trades = []
        positions = {}  # {ticker: position_dict}

        # Sort tweets by time
        tweets_df = tweets_df.sort_values('entry_time').reset_index(drop=True)

        # Score all tweets
        print("\nScoring tweets...")
        tweet_scores = {}  # {idx: {ticker: score}}
        tweet_meta = {}   # {idx: {'tweet_snippet': str, 'signal_reasons': {ticker: str}}}

        for idx, row in tweets_df.iterrows():
            scores = {}
            signal_reasons = {}
            for ticker in tickers:
                try:
                    score_dict = {}
                    if isinstance(scorer, type):
                        score = 0.0
                    else:
                        # Score tweet-ticker pair
                        if use_multi_timeframe and returns_df is not None and hasattr(scorer, 'score_tweet_ticker'):
                            try:
                                score_dict = scorer.score_tweet_ticker(row, ticker, returns_df)
                                score = score_dict.get('influence_score', 0.0)
                            except Exception:
                                try:
                                    score_dict = scorer.score_tweet_ticker(row, ticker, return_horizon=return_horizon)
                                    score = score_dict.get('influence_score', 0.0)
                                except:
                                    score = 0.0
                        elif hasattr(scorer, 'score_tweet_ticker'):
                            try:
                                score_dict = scorer.score_tweet_ticker(row, ticker, return_horizon=return_horizon)
                                score = score_dict.get('influence_score', 0.0)
                            except TypeError:
                                score_dict = scorer.score_tweet_ticker(row, ticker)
                                score = score_dict.get('influence_score', 0.0)
                        elif hasattr(scorer, 'predict_tweet_ticker'):
                            score = scorer.predict_tweet_ticker(row, ticker, return_horizon)
                        else:
                            try:
                                from .correlation_discovery import CorrelationDiscovery
                                from .bag_of_words_scorer import BagOfWordsScorer
                                from .embedding_scorer import EmbeddingScorer
                                from .llm_scorer import LLMScorer
                                from .ensemble_scorer import EnsembleScorer

                                if isinstance(scorer, CorrelationDiscovery):
                                    score_dict = scorer.score_tweet_ticker(row, ticker, return_horizon)
                                    score = score_dict.get('influence_score', 0.0)
                                elif isinstance(scorer, BagOfWordsScorer):
                                    tweet_text = row.get('tweet_content', '')
                                    if isinstance(tweet_text, pd.Series):
                                        tweet_text = tweet_text.iloc[0]
                                    if pd.notna(tweet_text):
                                        score_dict = scorer.score_tweet_ticker(str(tweet_text), ticker)
                                        score = score_dict.get('influence_score', 0.0)
                                    else:
                                        score = 0.0
                                elif isinstance(scorer, EmbeddingScorer):
                                    score_dict = scorer.score_tweet_ticker(row, ticker, None, return_horizon)
                                    score = score_dict.get('influence_score', 0.0)
                                elif isinstance(scorer, LLMScorer):
                                    tweet_text = row.get('tweet_content', '')
                                    if isinstance(tweet_text, pd.Series):
                                        tweet_text = tweet_text.iloc[0]
                                    if pd.notna(tweet_text):
                                        score_dict = scorer.score_tweet_ticker(str(tweet_text), ticker)
                                        score = score_dict.get('influence_score', 0.0)
                                    else:
                                        score = 0.0
                                elif isinstance(scorer, EnsembleScorer):
                                    score_dict = scorer.score_tweet_ticker(row, ticker, None, return_horizon)
                                    score = score_dict.get('influence_score', 0.0)
                                else:
                                    score = 0.0
                                reason = _build_reason(scorer, score_dict, ticker)
                            except Exception:
                                score = 0.0
                    scores[ticker] = score
                    signal_reasons[ticker] = _build_reason(scorer, score_dict, ticker)
                except Exception as e:
                    print(f"  Warning: Error scoring tweet {idx} for {ticker}: {e}")
                    scores[ticker] = 0.0
                    signal_reasons[ticker] = ''

            tweet_scores[idx] = scores
            # Store metadata for richer trade output
            raw_content = row.get('tweet_content', '')
            if isinstance(raw_content, pd.Series):
                raw_content = raw_content.iloc[0]
            snippet = str(raw_content)[:120] if pd.notna(raw_content) else ''
            tweet_meta[idx] = {'tweet_snippet': snippet, 'signal_reasons': signal_reasons}

        print(f"  Scored {len(tweet_scores)} tweets")

        # Backtest loop
        print("\nRunning backtest...")
        daily_trade_count = {}  # Track trades per day
        
        for idx, row in tweets_df.iterrows():
            current_time = row['entry_time']
            ticker = row['ticker']
            
            # Time-of-day filter
            if self.use_time_filter:
                hour = pd.to_datetime(current_time).hour
                if hour < self.min_hour or hour > self.max_hour:
                    continue
            
            # Daily trade limit
            date = pd.to_datetime(current_time).date()
            if date not in daily_trade_count:
                daily_trade_count[date] = {}
            if ticker not in daily_trade_count[date]:
                daily_trade_count[date][ticker] = 0
            
            if daily_trade_count[date][ticker] >= self.max_trades_per_day:
                continue

            # Check if we have an active position for this ticker
            if ticker in positions:
                pos = positions[ticker]
                pos_exit_time = pos['exit_time']

                # If current tweet is within holding period, skip
                if current_time < pos_exit_time:
                    continue
                else:
                    # Close existing position
                    if ticker in price_data:
                        price_df = price_data[ticker]
                        gross_return = self.return_calculator.calculate_return(
                            entry_time=pos['entry_time'],
                            exit_time=current_time,
                            ticker=ticker,
                            price_df=price_df
                        )
                    else:
                        # Fallback: use forward return if available
                        ret_col = f'{ticker}_{return_horizon}'
                        if ret_col in row:
                            gross_return = row[ret_col]
                        else:
                            gross_return = 0.0

                    # Apply transaction costs
                    if self.apply_transaction_costs:
                        net_return = self.cost_calculator.apply_costs_to_return(
                            gross_return, pos['weight']
                        )
                        transaction_cost = capital * pos['weight'] * (gross_return - net_return)
                    else:
                        net_return = gross_return
                        transaction_cost = 0.0

                    # Calculate P&L
                    pnl = capital * pos['weight'] * net_return
                    capital += pnl

                    # Update position sizer history
                    if self.use_advanced_sizing and self.position_sizer:
                        self.position_sizer.update_performance({
                            'return': net_return,
                            'ticker': ticker
                        })

                    duration_mins = (
                        pd.Timestamp(current_time) - pd.Timestamp(pos['entry_time'])
                    ).total_seconds() / 60

                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'duration_minutes': round(duration_mins, 1),
                        'ticker': ticker,
                        'direction': 'LONG' if pos['influence_score'] >= 0 else 'SHORT',
                        'position_size_usd': round(capital * pos['weight'], 2),
                        'weight': pos['weight'],
                        'influence_score': pos['influence_score'],
                        'signal_reason': pos.get('signal_reason', ''),
                        'tweet_snippet': pos.get('tweet_snippet', ''),
                        'gross_return': gross_return,
                        'net_return': net_return,
                        'transaction_cost': transaction_cost,
                        'pnl': pnl,
                        'capital': capital
                    })

                    equity_curve.append({
                        'time': current_time,
                        'equity': capital
                    })

                    del positions[ticker]

            # Get score for this tweet-ticker pair
            scores = tweet_scores.get(idx, {})
            influence_score = scores.get(ticker, 0.0)

            # Market regime detection and threshold adaptation
            adaptive_threshold = min_score_threshold
            regimes = {}
            if self.use_regime_detection and self.regime_detector:
                # Detect regime for each ticker
                for t in tickers:
                    if t in price_data:
                        regime = self.regime_detector.detect_regime(price_data, t)
                        regimes[t] = regime
                        # Adapt threshold based on regime
                        adaptive_threshold = self.regime_detector.adapt_threshold(
                            regime, min_score_threshold
                        )

            # Only take trades above threshold
            # Also filter by absolute value to avoid noise
            if abs(influence_score) >= abs(adaptive_threshold):
                # Get all tweets at current time
                current_tweets = tweets_df[tweets_df['entry_time'] == current_time]
                
                # Get scores for all current tweets
                current_scores = []
                for _, tweet_row in current_tweets.iterrows():
                    tweet_idx = tweet_row.name if 'name' in tweet_row else current_tweets.index[current_tweets.index.get_loc(tweet_row.name)]
                    tweet_scores_dict = tweet_scores.get(tweet_idx, {})
                    current_scores.append(tweet_scores_dict.get(ticker, 0.0))

                if len(current_scores) > 0:
                    # Calculate position size
                    if self.use_advanced_sizing and self.position_sizer:
                        # Get volatility for this ticker
                        if ticker in price_data:
                            price_df = price_data[ticker]
                            if 'close' in price_df.columns:
                                recent_returns = price_df['close'].pct_change().dropna()
                                if len(recent_returns) > 0:
                                    volatility = recent_returns.tail(20).std()
                                else:
                                    volatility = 0.01
                            else:
                                volatility = 0.01
                        else:
                            volatility = 0.01
                        
                        # Get confidence from scorer if available
                        try:
                            score_dict = scorer.score_tweet_ticker(row, ticker, return_horizon=return_horizon)
                            confidence = score_dict.get('confidence', 0.5)
                        except:
                            confidence = 0.5
                        
                        # Get trade statistics for Kelly
                        stats = self.position_sizer.get_statistics()
                        
                        # Calculate position size
                        base_weight = abs(influence_score) * 10  # Scale
                        weight = self.position_sizer.calculate_size(
                            influence_score=influence_score,
                            confidence=confidence,
                            volatility=volatility,
                            win_rate=stats['win_rate'],
                            avg_win=stats['avg_win'],
                            avg_loss=stats['avg_loss']
                        )
                        
                        # Adjust for regime if available
                        if ticker in regimes and self.regime_detector:
                            regime = regimes[ticker]
                            weight = self.regime_detector.adapt_position_size(regime, weight)
                    else:
                        # Use softmax to allocate capital (original method)
                        signals = np.array(current_scores)
                        weights = self.softmax(signals)
                        
                        # Find this tweet's index in current_tweets
                        tweet_idx_in_batch = current_tweets.index.get_loc(idx) if idx in current_tweets.index else 0
                        weight = weights[tweet_idx_in_batch] if tweet_idx_in_batch < len(weights) else 0.0

                    # Open position
                    exit_time = current_time + pd.Timedelta(minutes=self.holding_period_minutes)
                    meta = tweet_meta.get(idx, {})
                    positions[ticker] = {
                        'entry_time': current_time,
                        'exit_time': exit_time,
                        'weight': weight,
                        'influence_score': influence_score,
                        'tweet_idx': idx,
                        'signal_reason': meta.get('signal_reasons', {}).get(ticker, ''),
                        'tweet_snippet': meta.get('tweet_snippet', ''),
                    }

                    daily_trade_count[date][ticker] += 1

        # Close any remaining positions
        if len(tweets_df) > 0:
            last_time = tweets_df['entry_time'].max()
            for ticker, pos in positions.items():
                if ticker in price_data:
                    price_df = price_data[ticker]
                    gross_return = self.return_calculator.calculate_return(
                        entry_time=pos['entry_time'],
                        exit_time=last_time,
                        ticker=ticker,
                        price_df=price_df
                    )
                else:
                    gross_return = 0.0

                if self.apply_transaction_costs:
                    net_return = self.cost_calculator.apply_costs_to_return(
                        gross_return, pos['weight']
                    )
                    transaction_cost = self.initial_capital * pos['weight'] * (gross_return - net_return)
                else:
                    net_return = gross_return
                    transaction_cost = 0.0

                pnl = capital * pos['weight'] * net_return
                capital += pnl

                duration_mins = (
                    pd.Timestamp(last_time) - pd.Timestamp(pos['entry_time'])
                ).total_seconds() / 60

                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': last_time,
                    'duration_minutes': round(duration_mins, 1),
                    'ticker': ticker,
                    'direction': 'LONG' if pos['influence_score'] >= 0 else 'SHORT',
                    'position_size_usd': round(capital * pos['weight'], 2),
                    'weight': pos['weight'],
                    'influence_score': pos['influence_score'],
                    'signal_reason': pos.get('signal_reason', ''),
                    'tweet_snippet': pos.get('tweet_snippet', ''),
                    'gross_return': gross_return,
                    'net_return': net_return,
                    'transaction_cost': transaction_cost,
                    'pnl': pnl,
                    'capital': capital
                })

                equity_curve.append({
                    'time': last_time,
                    'equity': capital
                })

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()

        if len(equity_df) > 0:
            equity_df = equity_df.sort_values('time').reset_index(drop=True)
            # Ensure equity never goes below 0
            equity_df['equity'] = equity_df['equity'].clip(lower=0.01)

        return trades_df, equity_df, capital
