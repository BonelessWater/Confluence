"""
IMPROVED STRATEGY V3 - Based on Deep Analysis

Key improvements from analysis:
1. Asset-specific signals (not just SPY)
2. Use statistically significant keyword-asset pairs
3. Longer holding periods (30-60m)
4. Time filtering (12:00-14:00)
5. Keyword combinations for stronger signals
6. Relative value trades to reduce market exposure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# Statistically significant signals from analysis
SIGNALS = [
    # (keyword_pattern, ticker, direction, holding_mins, expected_return, t_stat)
    ('tariff', 'EWW', 1, 60, 0.0032, 3.06),      # BEST: Long EWW on tariff
    ('china', 'DIA', 1, 30, 0.0011, 2.59),       # Long DIA on china
    ('china', 'SPY', 1, 30, 0.0007, 2.26),       # Long SPY on china
    ('crash', 'TLT', 1, 60, 0.0017, 2.44),       # Long TLT on crash (flight to safety)
    ('weak', 'USO', -1, 30, 0.0015, 2.79),       # Short USO on weak
    ('mexico', 'QCOM', -1, 15, 0.0012, 2.85),    # Short QCOM on mexico
    ('great', 'QCOM', -1, 15, 0.0006, 2.64),     # Short QCOM on great (contrarian)
]

# Keyword combinations (stronger signals)
COMBO_SIGNALS = [
    (['tariff', 'china'], 'SPY', 1, 30, 0.0009),   # tariff+china -> long SPY
    (['tariff', 'trade'], 'SPY', 1, 30, 0.00125),  # tariff+trade -> long SPY
    (['china', 'trade'], 'SPY', 1, 30, 0.00129),   # china+trade -> long SPY
]


class ImprovedStrategyV3:
    """Improved strategy using asset-specific signals."""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission_bps: float = 5.0,
                 slippage_bps: float = 3.0,
                 max_trades_per_day: int = 2,
                 use_time_filter: bool = True,
                 min_t_stat: float = 2.0):

        self.initial_capital = initial_capital
        self.total_cost_bps = commission_bps + slippage_bps
        self.max_trades_per_day = max_trades_per_day
        self.use_time_filter = use_time_filter
        self.min_t_stat = min_t_stat

        # Filter signals by t-stat threshold
        self.signals = [s for s in SIGNALS if abs(s[5]) >= min_t_stat]

    def load_data(self):
        """Load tweet and market data."""
        print("Loading data...")

        # Load tweets
        features_path = Path(__file__).parent.parent / 'data' / 'trump-truth-social-archive' / 'data' / 'truth_archive_with_features.parquet'
        tweets = pd.read_parquet(features_path)
        tweets = tweets[['tweet_time', 'tweet_content', 'entry_time']].drop_duplicates(subset=['entry_time']).copy()

        if tweets['entry_time'].dt.tz is not None:
            tweets['entry_time'] = tweets['entry_time'].dt.tz_localize(None)

        tweets = tweets.sort_values('entry_time').reset_index(drop=True)

        # Load market data for all tickers in signals
        tickers_needed = set([s[1] for s in self.signals])
        market_data = {}

        for ticker in tickers_needed:
            path = Path(__file__).parent.parent / 'data' / f'{ticker}.parquet'
            if path.exists():
                market_data[ticker] = pl.read_parquet(path)
                print(f"  Loaded {ticker}")

        return tweets, market_data

    def prepare_data(self, tweets: pd.DataFrame, market_data: dict) -> pd.DataFrame:
        """Calculate returns for all tickers."""
        df = tweets.copy()

        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
        df['date'] = pd.to_datetime(df['entry_time']).dt.date
        df['content_lower'] = df['tweet_content'].fillna('').str.lower()

        for ticker, market_pl in market_data.items():
            min_time = df['entry_time'].min()
            max_time = df['entry_time'].max() + pd.Timedelta(hours=2)

            market_1min = (
                market_pl
                .with_columns(pl.col('ts_event').dt.replace_time_zone(None))
                .filter((pl.col('ts_event') >= min_time) & (pl.col('ts_event') <= max_time))
                .with_columns(pl.col('ts_event').dt.truncate('1m').alias('minute'))
                .group_by('minute')
                .agg([pl.col('close').last()])
                .sort('minute')
            )
            price_df = market_1min.to_pandas().set_index('minute')['close']

            entry_times = df['entry_time'].dt.floor('min')
            entry_prices = price_df.reindex(entry_times).values
            df[f'{ticker}_price'] = entry_prices

            for mins in [15, 30, 60]:
                exit_times = (df['entry_time'] + pd.Timedelta(minutes=mins)).dt.floor('min')
                exit_prices = price_df.reindex(exit_times).values
                df[f'{ticker}_{mins}m'] = (exit_prices - entry_prices) / entry_prices

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on keyword-asset mappings."""
        df = df.copy()

        # Initialize signal columns
        df['trade_ticker'] = ''
        df['trade_direction'] = 0
        df['trade_holding'] = 0
        df['trade_expected'] = 0.0
        df['trade_reason'] = ''

        for idx, row in df.iterrows():
            content = row['content_lower']
            hour = row['hour']

            # Time filter: only trade 12:00-14:00 (best hours)
            if self.use_time_filter and (hour < 12 or hour > 14):
                continue

            best_signal = None
            best_t_stat = 0

            # Check each signal
            for keyword, ticker, direction, holding, expected, t_stat in self.signals:
                if keyword in content:
                    # Check if we have the return data
                    ret_col = f'{ticker}_{holding}m'
                    if ret_col not in df.columns or pd.isna(row.get(ret_col)):
                        continue

                    # Select strongest signal (by t-stat)
                    if abs(t_stat) > abs(best_t_stat):
                        best_signal = (keyword, ticker, direction, holding, expected, t_stat)
                        best_t_stat = t_stat

            # Check combo signals (stronger)
            for keywords, ticker, direction, holding, expected in COMBO_SIGNALS:
                if all(kw in content for kw in keywords):
                    ret_col = f'{ticker}_{holding}m'
                    if ret_col not in df.columns or pd.isna(row.get(ret_col)):
                        continue

                    # Combos get priority (assume t_stat = 3.0)
                    if 3.0 > abs(best_t_stat):
                        best_signal = ('+'.join(keywords), ticker, direction, holding, expected, 3.0)
                        best_t_stat = 3.0

            if best_signal:
                keyword, ticker, direction, holding, expected, t_stat = best_signal
                df.at[idx, 'trade_ticker'] = ticker
                df.at[idx, 'trade_direction'] = direction
                df.at[idx, 'trade_holding'] = holding
                df.at[idx, 'trade_expected'] = expected
                df.at[idx, 'trade_reason'] = f"{keyword}->{ticker}"

        return df

    def backtest(self, df: pd.DataFrame) -> tuple:
        """Run backtest."""
        print("\nRunning backtest...")

        df = df.copy().sort_values('entry_time').reset_index(drop=True)

        capital = self.initial_capital
        trades = []
        equity_curve = [{'time': df['entry_time'].iloc[0], 'equity': capital}]

        daily_count = {}
        tradeable = df[df['trade_direction'] != 0]

        print(f"  Tradeable signals: {len(tradeable)}")

        for idx, row in tradeable.iterrows():
            date = row['date']
            if date not in daily_count:
                daily_count[date] = 0
            if daily_count[date] >= self.max_trades_per_day:
                continue

            ticker = row['trade_ticker']
            direction = row['trade_direction']
            holding = row['trade_holding']
            ret_col = f'{ticker}_{holding}m'

            if ret_col not in row or pd.isna(row[ret_col]):
                continue

            fwd_return = row[ret_col]
            gross_return = fwd_return * direction
            cost = self.total_cost_bps / 10000
            net_return = gross_return - cost

            pnl = capital * net_return
            capital += pnl

            trades.append({
                'entry_time': row['entry_time'],
                'tweet': row['tweet_content'][:80] if pd.notna(row['tweet_content']) else '',
                'ticker': ticker,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'holding': holding,
                'reason': row['trade_reason'],
                'fwd_return': fwd_return,
                'gross_return': gross_return,
                'net_return': net_return,
                'pnl': pnl,
                'capital': capital
            })

            equity_curve.append({'time': row['entry_time'], 'equity': capital})
            daily_count[date] += 1

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)

        return trades_df, equity_df, capital

    def calculate_metrics(self, trades_df: pd.DataFrame, final_capital: float):
        """Calculate and print metrics."""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)

        total_return = (final_capital - self.initial_capital) / self.initial_capital
        n_trades = len(trades_df)

        print(f"\nCapital: ${self.initial_capital:,.0f} -> ${final_capital:,.2f} ({total_return*100:+.2f}%)")
        print(f"Trades: {n_trades}")

        if n_trades > 0:
            win_rate = (trades_df['pnl'] > 0).sum() / n_trades
            avg_return = trades_df['net_return'].mean()

            print(f"Win rate: {win_rate*100:.1f}%")
            print(f"Avg return: {avg_return*100:.4f}%")

            if n_trades > 1:
                sharpe = np.mean(trades_df['net_return']) / np.std(trades_df['net_return']) * np.sqrt(252)
                print(f"Sharpe: {sharpe:.2f}")

            # By ticker
            print("\nBy Ticker:")
            for ticker in trades_df['ticker'].unique():
                t_trades = trades_df[trades_df['ticker'] == ticker]
                t_ret = t_trades['net_return'].mean() * 100
                t_win = (t_trades['pnl'] > 0).mean() * 100
                print(f"  {ticker}: {len(t_trades)} trades, {t_win:.1f}% win, {t_ret:+.4f}% avg")

            # By signal reason
            print("\nBy Signal:")
            for reason in trades_df['reason'].unique():
                r_trades = trades_df[trades_df['reason'] == reason]
                r_ret = r_trades['net_return'].mean() * 100
                r_win = (r_trades['pnl'] > 0).mean() * 100
                print(f"  {reason}: {len(r_trades)} trades, {r_win:.1f}% win, {r_ret:+.4f}% avg")


def main():
    print("="*70)
    print("IMPROVED STRATEGY V3 - Asset-Specific Signals")
    print("="*70)
    print("\nUsing statistically significant signals:")
    for kw, ticker, direction, holding, expected, t_stat in SIGNALS:
        dir_str = "LONG" if direction == 1 else "SHORT"
        print(f"  {kw} -> {dir_str} {ticker} ({holding}m), t={t_stat:.2f}")

    strategy = ImprovedStrategyV3(
        initial_capital=100000,
        max_trades_per_day=2,
        use_time_filter=True,
        min_t_stat=2.0
    )

    # Load data
    tweets, market_data = strategy.load_data()
    print(f"\nTweets: {len(tweets)}")

    # Prepare data
    df = strategy.prepare_data(tweets, market_data)

    # Train/test split
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

    # Generate signals on test set
    test_df = strategy.generate_signals(test_df)

    # Backtest
    trades_df, equity_df, final_capital = strategy.backtest(test_df)

    # Metrics
    strategy.calculate_metrics(trades_df, final_capital)

    # Save
    if len(trades_df) > 0:
        trades_df.to_csv('output/improved_v3_trades.csv', index=False)
        print(f"\n[OK] Trades saved to: output/improved_v3_trades.csv")

    # Compare with no time filter
    print("\n" + "="*70)
    print("COMPARISON: With vs Without Time Filter")
    print("="*70)

    strategy_nofilter = ImprovedStrategyV3(
        initial_capital=100000,
        max_trades_per_day=2,
        use_time_filter=False,
        min_t_stat=2.0
    )

    test_df_nf = strategy_nofilter.generate_signals(df.iloc[split_idx:].copy())
    trades_nf, _, capital_nf = strategy_nofilter.backtest(test_df_nf)

    ret_filter = (final_capital - 100000) / 100000 * 100
    ret_nofilter = (capital_nf - 100000) / 100000 * 100

    print(f"\nWith time filter (12-14): {len(trades_df)} trades, {ret_filter:+.2f}%")
    print(f"Without time filter:      {len(trades_nf)} trades, {ret_nofilter:+.2f}%")


if __name__ == "__main__":
    main()
