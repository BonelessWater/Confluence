"""
FINAL ROBUST STRATEGY - Honest Assessment

KEY FINDING: With proper statistical controls (Bonferroni/BH correction),
no keywords show significant predictive power for SPY returns.

This script shows THREE approaches:
1. RIGOROUS: Full multiple testing correction → No trades (most honest)
2. MODERATE: Single best keyword per holding period → Some trades
3. AGGRESSIVE: Top signals without correction → More trades (highest overfit risk)

The user must decide their risk tolerance for overfitting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class FinalRobustStrategy:
    """Final robust strategy with multiple risk levels."""

    def __init__(self, initial_capital: float = 100000,
                 commission_bps: float = 5.0, slippage_bps: float = 3.0,
                 max_trades_per_day: int = 2):

        self.initial_capital = initial_capital
        self.total_cost_bps = commission_bps + slippage_bps
        self.max_trades_per_day = max_trades_per_day

        self.candidate_keywords = [
            'tariff', 'tariffs', 'trade', 'china', 'jobs', 'economy',
            'stock', 'market', 'dow', 'crash', 'great', 'strong', 'weak'
        ]

    def load_and_prepare(self, ticker: str = 'SPY') -> pd.DataFrame:
        """Load and prepare all data."""
        print(f"\nLoading {ticker}...")

        features_path = Path(__file__).parent.parent / 'data' / 'trump-truth-social-archive' / 'data' / 'truth_archive_with_features.parquet'
        df = pd.read_parquet(features_path)
        df = df[df['ticker'] == ticker].copy()

        market_path = Path(__file__).parent.parent / 'data' / f'{ticker}.parquet'
        market_pl = pl.read_parquet(market_path)

        if df['entry_time'].dt.tz is not None:
            df['entry_time'] = df['entry_time'].dt.tz_localize(None)

        min_time = df['entry_time'].min()
        max_time = df['entry_time'].max() + pd.Timedelta(hours=2)

        market_1min = (
            market_pl
            .with_columns(pl.col('ts_event').dt.replace_time_zone(None))
            .filter((pl.col('ts_event') >= min_time) & (pl.col('ts_event') <= max_time))
            .with_columns(pl.col('ts_event').dt.truncate('1m').alias('minute'))
            .group_by('minute').agg([pl.col('close').last()])
            .sort('minute')
        )
        price_df = market_1min.to_pandas().set_index('minute')['close']

        entry_times = df['entry_time'].dt.floor('min')
        entry_prices = price_df.reindex(entry_times).values
        df['entry_price'] = entry_prices

        for mins in [5, 15, 30, 60]:
            exit_times = (df['entry_time'] + pd.Timedelta(minutes=mins)).dt.floor('min')
            exit_prices = price_df.reindex(exit_times).values
            df[f'return_{mins}m'] = (exit_prices - entry_prices) / entry_prices
            df[f'return_{mins}m'] = df[f'return_{mins}m'].fillna(0)

        for mins in [15, 30]:
            past_times = (df['entry_time'] - pd.Timedelta(minutes=mins)).dt.floor('min')
            past_prices = price_df.reindex(past_times).values
            df[f'past_{mins}m'] = (entry_prices - past_prices) / past_prices
            df[f'past_{mins}m'] = df[f'past_{mins}m'].fillna(0)

        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
        df['market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['date'] = pd.to_datetime(df['entry_time']).dt.date
        df['content_lower'] = df['tweet_content'].fillna('').str.lower()
        df['realized_vol'] = df['past_30m'].abs()

        df = df[df['return_5m'] != 0].copy()
        print(f"  Valid samples: {len(df)}")

        return df

    def analyze_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze all keyword/holding combinations."""
        results = []

        for holding in [5, 15, 30, 60]:
            ret_col = f'return_{holding}m'

            for keyword in self.candidate_keywords:
                mask = df['content_lower'].str.contains(keyword, na=False) & (df[ret_col] != 0)
                n = mask.sum()

                if n >= 20:
                    returns = df.loc[mask, ret_col].values
                    mean_ret = np.mean(returns)
                    std_ret = np.std(returns, ddof=1)
                    se = std_ret / np.sqrt(n)
                    t_stat, p_value = stats.ttest_1samp(returns, 0)

                    results.append({
                        'keyword': keyword,
                        'holding': holding,
                        'n': n,
                        'mean_return': mean_ret,
                        'std_return': std_ret,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'ci_lower': mean_ret - 1.96 * se,
                        'ci_upper': mean_ret + 1.96 * se,
                        'win_rate': (returns > 0).mean()
                    })

        results_df = pd.DataFrame(results)

        # Multiple testing corrections
        n_tests = len(results_df)
        results_df['p_bonferroni'] = np.minimum(results_df['p_value'] * n_tests, 1.0)

        # Benjamini-Hochberg
        p_sorted_idx = np.argsort(results_df['p_value'].values)
        p_sorted = results_df['p_value'].values[p_sorted_idx]
        bh_adj = np.zeros(n_tests)
        for i, orig_idx in enumerate(p_sorted_idx):
            bh_adj[orig_idx] = p_sorted[i] * n_tests / (i + 1)
        results_df['p_bh'] = np.minimum(bh_adj, 1.0)

        return results_df

    def backtest_signals(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                        signals: List[dict], name: str) -> Tuple[pd.DataFrame, float]:
        """Backtest a set of signals."""
        test_df = test_df.copy()
        test_df['signal'] = 0
        test_df['signal_keyword'] = ''
        test_df['signal_holding'] = 0

        for sig in signals:
            keyword = sig['keyword']
            holding = sig['holding']
            direction = 1 if sig['mean_return'] > 0 else -1

            mask = (
                test_df['content_lower'].str.contains(keyword, na=False) &
                (test_df['market_hours'] == 1)
            )

            # Only update if this signal is stronger
            stronger = mask & (abs(sig['t_stat']) > test_df['signal'].abs())
            test_df.loc[stronger, 'signal'] = direction
            test_df.loc[stronger, 'signal_keyword'] = keyword
            test_df.loc[stronger, 'signal_holding'] = holding

        # Volatility filter
        vol_75 = train_df['realized_vol'].quantile(0.75)
        test_df.loc[test_df['realized_vol'] > vol_75, 'signal'] = 0

        # Execute
        capital = self.initial_capital
        trades = []
        daily_count = {}

        tradeable = test_df[test_df['signal'] != 0]

        for idx, row in tradeable.iterrows():
            date = row['date']
            if date not in daily_count:
                daily_count[date] = 0
            if daily_count[date] >= self.max_trades_per_day:
                continue

            holding = row['signal_holding']
            ret_col = f'return_{holding}m'
            if ret_col not in row or pd.isna(row[ret_col]):
                continue

            fwd_ret = row[ret_col]
            direction = row['signal']

            gross = fwd_ret * direction
            cost = self.total_cost_bps / 10000
            net = gross - cost

            pnl = capital * net
            capital += pnl

            trades.append({
                'entry_time': row['entry_time'],
                'keyword': row['signal_keyword'],
                'holding': holding,
                'direction': direction,
                'gross_return': gross,
                'net_return': net,
                'pnl': pnl,
                'capital': capital
            })

            daily_count[date] += 1

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return trades_df, capital

    def run_comparison(self, ticker: str = 'SPY'):
        """Run comparison of different risk levels."""
        print("="*70)
        print("FINAL ROBUST STRATEGY - Honest Comparison")
        print("="*70)

        df = self.load_and_prepare(ticker)

        # Split
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

        # Analyze on training
        print("\n" + "="*60)
        print("SIGNAL ANALYSIS (Training Data)")
        print("="*60)

        stats_df = self.analyze_all_signals(train_df)
        stats_df = stats_df.sort_values('t_stat', ascending=False)

        print("\nTop 10 signals by t-statistic:")
        display_cols = ['keyword', 'holding', 'n', 'mean_return', 't_stat', 'p_value', 'p_bonferroni', 'p_bh']
        top10 = stats_df.head(10)[display_cols].copy()
        top10['mean_return'] = top10['mean_return'].apply(lambda x: f'{x*100:.4f}%')
        print(top10.to_string(index=False))

        # Count significant at different levels
        print("\n" + "-"*40)
        print("Significance counts:")
        print(f"  p < 0.05 (raw):        {(stats_df['p_value'] < 0.05).sum()}")
        print(f"  p < 0.10 (raw):        {(stats_df['p_value'] < 0.10).sum()}")
        print(f"  p < 0.10 (Bonferroni): {(stats_df['p_bonferroni'] < 0.10).sum()}")
        print(f"  p < 0.20 (BH FDR):     {(stats_df['p_bh'] < 0.20).sum()}")

        # Define three approaches
        approaches = {}

        # 1. RIGOROUS: Bonferroni corrected at 10%
        rigorous_signals = stats_df[stats_df['p_bonferroni'] < 0.10].to_dict('records')
        approaches['RIGOROUS (Bonferroni)'] = rigorous_signals

        # 2. MODERATE: BH corrected at 20%
        moderate_signals = stats_df[stats_df['p_bh'] < 0.20].to_dict('records')
        approaches['MODERATE (BH FDR)'] = moderate_signals

        # 3. AGGRESSIVE: Raw p < 0.10, top 3
        aggressive_signals = stats_df[stats_df['p_value'] < 0.10].head(3).to_dict('records')
        approaches['AGGRESSIVE (Top 3 raw)'] = aggressive_signals

        # Run backtests
        print("\n" + "="*60)
        print("OUT-OF-SAMPLE BACKTEST RESULTS")
        print("="*60)

        results = {}
        for name, signals in approaches.items():
            print(f"\n{name}:")
            print(f"  Signals: {len(signals)}")

            if len(signals) > 0:
                for s in signals[:3]:
                    print(f"    - {s['keyword']} ({s['holding']}m): t={s['t_stat']:.2f}, p={s['p_value']:.4f}")

            trades_df, final_capital = self.backtest_signals(train_df, test_df, signals, name)

            total_return = (final_capital - self.initial_capital) / self.initial_capital
            n_trades = len(trades_df)

            print(f"  Trades: {n_trades}")
            print(f"  Return: {total_return*100:+.2f}%")
            print(f"  Final:  ${final_capital:,.2f}")

            if n_trades > 0:
                win_rate = (trades_df['pnl'] > 0).sum() / n_trades
                avg_ret = trades_df['net_return'].mean()
                print(f"  Win Rate: {win_rate*100:.1f}%")
                print(f"  Avg Return: {avg_ret*100:.4f}%")

                # Save trades
                safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
                trades_df.to_csv(f'output/{ticker}_{safe_name}_trades.csv', index=False)

            results[name] = {
                'signals': len(signals),
                'trades': n_trades,
                'return': total_return,
                'final': final_capital
            }

        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        print(f"\n{'Approach':<30} {'Signals':<10} {'Trades':<10} {'Return':<12} {'Final'}")
        print("-"*75)
        for name, res in results.items():
            print(f"{name:<30} {res['signals']:<10} {res['trades']:<10} {res['return']*100:+.2f}%{'':<6} ${res['final']:,.2f}")

        # Honest assessment
        print("\n" + "="*60)
        print("HONEST ASSESSMENT")
        print("="*60)
        print("""
With proper statistical controls, this dataset does NOT contain
reliable alpha that survives multiple testing correction.

The apparent patterns are likely due to:
1. Multiple testing (many keywords x holding periods)
2. Small effect sizes (~0.01-0.04% per trade)
3. High transaction costs (0.08% per trade) that exceed the edge

RECOMMENDATIONS:
- If you want statistical rigor: Accept that there's no tradeable signal
- If you accept overfit risk: Use the AGGRESSIVE approach, but expect
  out-of-sample performance to degrade
- Consider: Longer holding periods, lower transaction costs, or
  combining this signal with other alpha sources
""")

        # Save analysis
        stats_df.to_csv(f'output/{ticker}_full_signal_analysis.csv', index=False)
        print(f"\n[OK] Full analysis saved to: output/{ticker}_full_signal_analysis.csv")


def main():
    strategy = FinalRobustStrategy(
        initial_capital=100000,
        max_trades_per_day=2
    )
    strategy.run_comparison('SPY')


if __name__ == "__main__":
    main()
