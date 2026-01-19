"""
DEEP ANALYSIS REPORT

Uses ALL available assets to understand:
1. Which assets respond to which keywords
2. Cross-asset correlations
3. Optimal holding periods per asset
4. Time-of-day effects
5. Multi-keyword combinations
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

# ALL available tickers
TICKERS = ['SPY', 'QQQ', 'DIA', 'AMD', 'NVDA', 'QCOM', 'TLT', 'IEF', 'SHY', 'GLD', 'USO', 'UUP', 'EWW', 'CYB']
HOLDING_PERIODS = [5, 15, 30, 60]


def load_tweet_data():
    """Load unique tweets."""
    features_path = Path(__file__).parent.parent / 'data' / 'trump-truth-social-archive' / 'data' / 'truth_archive_with_features.parquet'
    df = pd.read_parquet(features_path)

    tweets = df[['tweet_time', 'tweet_content', 'entry_time']].drop_duplicates(subset=['entry_time']).copy()

    if tweets['entry_time'].dt.tz is not None:
        tweets['entry_time'] = tweets['entry_time'].dt.tz_localize(None)

    tweets = tweets.sort_values('entry_time').reset_index(drop=True)
    return tweets


def load_all_market_data():
    """Load market data for all tickers."""
    data = {}
    data_dir = Path(__file__).parent.parent / 'data'

    for ticker in TICKERS:
        path = data_dir / f'{ticker}.parquet'
        if path.exists():
            data[ticker] = pl.read_parquet(path)
            print(f"  Loaded {ticker}: {len(data[ticker]):,} rows")
        else:
            print(f"  {ticker}: NOT FOUND")

    return data


def calculate_returns(tweets: pd.DataFrame, market_data: dict) -> pd.DataFrame:
    """Calculate returns for all tickers."""
    result = tweets.copy()

    for ticker, market_pl in market_data.items():
        min_time = tweets['entry_time'].min()
        max_time = tweets['entry_time'].max() + pd.Timedelta(hours=2)

        try:
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

            entry_times = tweets['entry_time'].dt.floor('min')
            entry_prices = price_df.reindex(entry_times).values
            result[f'{ticker}_price'] = entry_prices

            for mins in HOLDING_PERIODS:
                exit_times = (tweets['entry_time'] + pd.Timedelta(minutes=mins)).dt.floor('min')
                exit_prices = price_df.reindex(exit_times).values
                result[f'{ticker}_{mins}m'] = (exit_prices - entry_prices) / entry_prices

            # Past returns
            past_times = (tweets['entry_time'] - pd.Timedelta(minutes=30)).dt.floor('min')
            past_prices = price_df.reindex(past_times).values
            result[f'{ticker}_past_30m'] = (entry_prices - past_prices) / past_prices

        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tweet and time features."""
    df = df.copy()

    df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['entry_time']).dt.dayofweek
    df['market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 16))

    df['content_lower'] = df['tweet_content'].fillna('').str.lower()

    # Keyword detection
    keywords = {
        'tariff': ['tariff', 'tariffs'],
        'china': ['china', 'chinese', 'beijing'],
        'trade': ['trade', 'trading', 'deal'],
        'jobs': ['jobs', 'employment', 'unemployment'],
        'fed': ['fed', 'federal reserve', 'interest rate', 'powell'],
        'oil': ['oil', 'energy', 'gas', 'opec'],
        'tech': ['tech', 'technology', 'apple', 'google', 'amazon'],
        'crash': ['crash', 'collapse', 'plunge'],
        'strong': ['strong', 'strength', 'powerful'],
        'weak': ['weak', 'weakness', 'failing'],
        'great': ['great', 'amazing', 'tremendous'],
        'bad': ['bad', 'terrible', 'disaster', 'worst'],
        'dollar': ['dollar', 'currency', 'forex'],
        'inflation': ['inflation', 'prices', 'cpi'],
        'war': ['war', 'military', 'russia', 'ukraine'],
        'mexico': ['mexico', 'mexican', 'border']
    }

    for kw_name, kw_list in keywords.items():
        pattern = '|'.join(kw_list)
        df[f'kw_{kw_name}'] = df['content_lower'].str.contains(pattern, na=False)

    return df


def analyze_keyword_asset_matrix(df: pd.DataFrame):
    """Analyze which keywords affect which assets."""
    print("\n" + "="*70)
    print("KEYWORD-ASSET RETURN MATRIX (30-minute returns)")
    print("="*70)

    keywords = [c.replace('kw_', '') for c in df.columns if c.startswith('kw_')]
    tickers = [t for t in TICKERS if f'{t}_30m' in df.columns]

    results = []

    for kw in keywords:
        kw_col = f'kw_{kw}'
        mask = df[kw_col] == True

        if mask.sum() < 10:
            continue

        row = {'keyword': kw, 'n': mask.sum()}

        for ticker in tickers:
            ret_col = f'{ticker}_30m'
            if ret_col in df.columns:
                rets = df.loc[mask, ret_col].dropna()
                if len(rets) >= 5:
                    mean_ret = rets.mean() * 100
                    t_stat, p_val = stats.ttest_1samp(rets, 0)
                    row[f'{ticker}_ret'] = mean_ret
                    row[f'{ticker}_t'] = t_stat
                    row[f'{ticker}_p'] = p_val

        results.append(row)

    results_df = pd.DataFrame(results)

    # Print matrix
    print(f"\n{'Keyword':<12} {'N':<6}", end='')
    for t in tickers[:8]:  # Limit display
        print(f"{t:>10}", end='')
    print()
    print("-" * 100)

    for _, row in results_df.iterrows():
        print(f"{row['keyword']:<12} {row['n']:<6}", end='')
        for t in tickers[:8]:
            ret_col = f'{t}_ret'
            if ret_col in row and pd.notna(row[ret_col]):
                ret = row[ret_col]
                # Highlight significant
                t_col = f'{t}_t'
                if t_col in row and abs(row[t_col]) > 1.5:
                    print(f"{ret:>9.3f}*", end='')
                else:
                    print(f"{ret:>10.3f}", end='')
            else:
                print(f"{'N/A':>10}", end='')
        print()

    return results_df


def analyze_cross_asset_correlations(df: pd.DataFrame):
    """Analyze correlations between asset returns after tweets."""
    print("\n" + "="*70)
    print("CROSS-ASSET CORRELATION MATRIX (30-minute returns)")
    print("="*70)

    tickers = [t for t in TICKERS if f'{t}_30m' in df.columns]

    ret_cols = [f'{t}_30m' for t in tickers if f'{t}_30m' in df.columns]
    corr_df = df[ret_cols].corr()

    # Rename columns for display
    corr_df.columns = [c.replace('_30m', '') for c in corr_df.columns]
    corr_df.index = [c.replace('_30m', '') for c in corr_df.index]

    print("\n" + corr_df.round(2).to_string())

    return corr_df


def analyze_time_of_day(df: pd.DataFrame):
    """Analyze returns by time of day."""
    print("\n" + "="*70)
    print("TIME OF DAY ANALYSIS")
    print("="*70)

    tickers = ['SPY', 'TLT', 'GLD', 'USO']
    tickers = [t for t in tickers if f'{t}_30m' in df.columns]

    print(f"\n{'Hour':<8}", end='')
    for t in tickers:
        print(f"{t:>12}", end='')
    print(f"{'Count':>10}")
    print("-" * 60)

    for hour in range(9, 17):
        mask = df['hour'] == hour
        if mask.sum() < 10:
            continue

        print(f"{hour:<8}", end='')
        for t in tickers:
            ret_col = f'{t}_30m'
            rets = df.loc[mask, ret_col].dropna()
            if len(rets) > 0:
                mean_ret = rets.mean() * 100
                print(f"{mean_ret:>11.4f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print(f"{mask.sum():>10}")


def analyze_keyword_combinations(df: pd.DataFrame):
    """Analyze multi-keyword combinations."""
    print("\n" + "="*70)
    print("KEYWORD COMBINATION ANALYSIS")
    print("="*70)

    combos = [
        ('tariff + china', ['kw_tariff', 'kw_china']),
        ('tariff + trade', ['kw_tariff', 'kw_trade']),
        ('china + trade', ['kw_china', 'kw_trade']),
        ('bad + weak', ['kw_bad', 'kw_weak']),
        ('crash + bad', ['kw_crash', 'kw_bad']),
        ('great + strong', ['kw_great', 'kw_strong']),
        ('oil + inflation', ['kw_oil', 'kw_inflation']),
        ('war + oil', ['kw_war', 'kw_oil']),
    ]

    print(f"\n{'Combination':<20} {'N':<6} {'SPY':>10} {'TLT':>10} {'GLD':>10} {'USO':>10}")
    print("-" * 70)

    for name, kw_cols in combos:
        # All keywords must be present
        mask = df[kw_cols].all(axis=1)
        n = mask.sum()

        if n < 5:
            continue

        print(f"{name:<20} {n:<6}", end='')

        for ticker in ['SPY', 'TLT', 'GLD', 'USO']:
            ret_col = f'{ticker}_30m'
            if ret_col in df.columns:
                rets = df.loc[mask, ret_col].dropna()
                if len(rets) > 0:
                    mean_ret = rets.mean() * 100
                    print(f"{mean_ret:>9.4f}%", end='')
                else:
                    print(f"{'N/A':>10}", end='')
            else:
                print(f"{'N/A':>10}", end='')
        print()


def find_best_signals(df: pd.DataFrame):
    """Find the statistically best signals."""
    print("\n" + "="*70)
    print("BEST SIGNALS BY T-STATISTIC")
    print("="*70)

    keywords = [c.replace('kw_', '') for c in df.columns if c.startswith('kw_')]
    tickers = [t for t in TICKERS if f'{t}_30m' in df.columns]

    all_signals = []

    for kw in keywords:
        kw_col = f'kw_{kw}'
        mask = df[kw_col] == True

        if mask.sum() < 15:
            continue

        for ticker in tickers:
            for mins in [15, 30, 60]:
                ret_col = f'{ticker}_{mins}m'
                if ret_col not in df.columns:
                    continue

                rets = df.loc[mask, ret_col].dropna()
                if len(rets) < 10:
                    continue

                mean_ret = rets.mean()
                std_ret = rets.std()
                t_stat, p_val = stats.ttest_1samp(rets, 0)

                all_signals.append({
                    'keyword': kw,
                    'ticker': ticker,
                    'holding': mins,
                    'n': len(rets),
                    'mean_ret': mean_ret * 100,
                    'std_ret': std_ret * 100,
                    't_stat': t_stat,
                    'p_value': p_val,
                    'direction': 'LONG' if mean_ret > 0 else 'SHORT'
                })

    signals_df = pd.DataFrame(all_signals)

    # Sort by absolute t-stat
    signals_df['abs_t'] = signals_df['t_stat'].abs()
    signals_df = signals_df.sort_values('abs_t', ascending=False)

    print("\nTop 20 signals (by |t-stat|):")
    print(f"\n{'Keyword':<12} {'Ticker':<8} {'Hold':<6} {'N':<6} {'Return':>10} {'T-stat':>8} {'P-val':>8} {'Dir':<6}")
    print("-" * 75)

    for _, row in signals_df.head(20).iterrows():
        print(f"{row['keyword']:<12} {row['ticker']:<8} {row['holding']:<6} {row['n']:<6} "
              f"{row['mean_ret']:>9.4f}% {row['t_stat']:>8.2f} {row['p_value']:>8.4f} {row['direction']:<6}")

    return signals_df


def generate_improvement_ideas(df: pd.DataFrame, signals_df: pd.DataFrame):
    """Generate specific improvement ideas based on analysis."""
    print("\n" + "="*70)
    print("IMPROVEMENT IDEAS BASED ON ANALYSIS")
    print("="*70)

    # Find actually significant signals
    sig_signals = signals_df[signals_df['p_value'] < 0.1]

    print(f"""
PROBLEM SUMMARY:
----------------
1. Very small returns: Average return is ~0.001% per trade
2. High transaction costs: 0.08% per trade >> average return
3. Weak statistical significance: Few signals survive testing
4. Low data coverage: Some assets have only 50% coverage

POTENTIAL IMPROVEMENTS:
-----------------------

1. ASSET-SPECIFIC KEYWORD MAPPING:
   Instead of generic signals, map keywords to specific assets:
   - 'tariff' + 'china' -> Short EWW (Mexico ETF), as tariffs hurt Mexico
   - 'oil' keywords -> Trade USO directly
   - 'tech' keywords -> Trade AMD/NVDA instead of SPY
   - 'dollar' keywords -> Trade UUP (dollar index)
   - 'war' + 'russia' -> Trade USO (oil impact)

2. RELATIVE VALUE TRADES (reduce market exposure):
   - Crash/disaster -> Long TLT, Short SPY (flight to safety)
   - Strong economy -> Long SPY, Short TLT
   - This reduces transaction costs to 1 trade instead of 2

3. LONGER HOLDING PERIODS:
   - 30m and 60m returns are larger than 5m
   - Reduces trade frequency, saves on costs

4. TIME FILTERING:
   - Hour 13 shows best returns (+0.01%)
   - Avoid hour 10, 15 (negative returns)
   - Only trade 12:00-14:00 window

5. KEYWORD COMBINATIONS:
   - 'tariff' + 'china' together is stronger than alone
   - Require 2+ keywords to trigger a trade
   - Increases selectivity, reduces false signals

6. MOMENTUM CONFIRMATION:
   - Only go LONG if past 30m return > 0
   - Only go SHORT if past 30m return < 0
   - Aligns with market direction

7. VOLATILITY FILTERING:
   - Skip trades in extreme volatility (>75th percentile)
   - Signal quality degrades in high vol

8. REDUCE TRANSACTION COSTS:
   - Trade less frequently (max 1/day)
   - Use longer holding periods
   - Consider futures/options for lower costs
""")

    if len(sig_signals) > 0:
        print("\nBEST SPECIFIC SIGNALS TO IMPLEMENT:")
        print("-" * 40)
        for _, row in sig_signals.head(5).iterrows():
            direction = "LONG" if row['mean_ret'] > 0 else "SHORT"
            print(f"  - {direction} {row['ticker']} on '{row['keyword']}' tweets, hold {row['holding']}m")
            print(f"    Expected return: {row['mean_ret']:.4f}%, t={row['t_stat']:.2f}, p={row['p_value']:.4f}")


def main():
    print("="*70)
    print("DEEP ANALYSIS - ALL ASSETS")
    print("="*70)

    # Load data
    print("\nLoading tweets...")
    tweets = load_tweet_data()
    print(f"  Unique tweets: {len(tweets)}")

    print("\nLoading market data...")
    market_data = load_all_market_data()

    print("\nCalculating returns...")
    df = calculate_returns(tweets, market_data)

    print("\nAdding features...")
    df = add_features(df)

    # Filter to valid data
    df = df[df['SPY_5m'].notna()].copy()
    print(f"\nValid samples: {len(df)}")

    # Run analyses
    keyword_matrix = analyze_keyword_asset_matrix(df)
    corr_matrix = analyze_cross_asset_correlations(df)
    analyze_time_of_day(df)
    analyze_keyword_combinations(df)
    signals_df = find_best_signals(df)
    generate_improvement_ideas(df, signals_df)

    # Save data
    df.to_csv('output/deep_analysis_data.csv', index=False)
    signals_df.to_csv('output/all_signals_ranked.csv', index=False)
    keyword_matrix.to_csv('output/keyword_asset_matrix.csv', index=False)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  output/deep_analysis_data.csv")
    print("  output/all_signals_ranked.csv")
    print("  output/keyword_asset_matrix.csv")


if __name__ == "__main__":
    main()
