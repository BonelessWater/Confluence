"""
MULTI-ASSET TWEET REPORT

Generates a report showing each tweet with:
- Full tweet content
- Market conditions for ALL assets (SPY, QQQ, DIA, IWM, TLT, GLD)
- Forward returns at multiple timeframes for each asset
- Signal generated (if any)
- Keywords detected

This helps understand how tweets affected different asset classes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


TICKERS = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT', 'GLD']
HOLDING_PERIODS = [5, 15, 30, 60]


def load_tweet_data():
    """Load raw tweet data."""
    print("Loading tweet data...")

    features_path = Path(__file__).parent.parent / 'data' / 'trump-truth-social-archive' / 'data' / 'truth_archive_with_features.parquet'
    df = pd.read_parquet(features_path)

    # Get unique tweets (they're duplicated per ticker)
    tweet_cols = ['tweet_time', 'tweet_content', 'entry_time']
    if 'tweet_id' in df.columns:
        tweet_cols.append('tweet_id')

    tweets = df[tweet_cols].drop_duplicates(subset=['entry_time']).copy()

    if tweets['entry_time'].dt.tz is not None:
        tweets['entry_time'] = tweets['entry_time'].dt.tz_localize(None)

    tweets = tweets.sort_values('entry_time').reset_index(drop=True)

    print(f"  Unique tweets: {len(tweets)}")
    return tweets


def load_market_data(ticker: str):
    """Load market data for a ticker."""
    market_path = Path(__file__).parent.parent / 'data' / f'{ticker}.parquet'

    if not market_path.exists():
        print(f"  Warning: {ticker} data not found")
        return None

    market_pl = pl.read_parquet(market_path)
    return market_pl


def calculate_returns_for_ticker(tweets: pd.DataFrame, market_pl: pl.DataFrame, ticker: str) -> pd.DataFrame:
    """Calculate returns for a single ticker."""

    min_time = tweets['entry_time'].min()
    max_time = tweets['entry_time'].max() + pd.Timedelta(hours=2)

    # Resample to 1-min bars
    market_1min = (
        market_pl
        .with_columns(pl.col('ts_event').dt.replace_time_zone(None))
        .filter((pl.col('ts_event') >= min_time) & (pl.col('ts_event') <= max_time))
        .with_columns(pl.col('ts_event').dt.truncate('1m').alias('minute'))
        .group_by('minute')
        .agg([
            pl.col('open').first(),
            pl.col('high').max(),
            pl.col('low').min(),
            pl.col('close').last()
        ])
        .sort('minute')
    )

    price_df = market_1min.to_pandas().set_index('minute')

    entry_times = tweets['entry_time'].dt.floor('min')
    entry_prices = price_df['close'].reindex(entry_times).values

    result = pd.DataFrame()
    result[f'{ticker}_price'] = entry_prices

    # Calculate returns for each holding period
    for mins in HOLDING_PERIODS:
        exit_times = (tweets['entry_time'] + pd.Timedelta(minutes=mins)).dt.floor('min')
        exit_prices = price_df['close'].reindex(exit_times).values
        returns = (exit_prices - entry_prices) / entry_prices
        result[f'{ticker}_{mins}m'] = returns

    # Past returns (context)
    for mins in [15, 30]:
        past_times = (tweets['entry_time'] - pd.Timedelta(minutes=mins)).dt.floor('min')
        past_prices = price_df['close'].reindex(past_times).values
        past_returns = (entry_prices - past_prices) / past_prices
        result[f'{ticker}_past_{mins}m'] = past_returns

    return result


def detect_keywords(content: str) -> dict:
    """Detect keywords in tweet content."""
    content_lower = str(content).lower() if pd.notna(content) else ''

    keyword_categories = {
        'bullish': ['tariff', 'tariffs', 'trade', 'china', 'jobs', 'growth', 'strong', 'great', 'winning', 'record'],
        'bearish': ['crash', 'disaster', 'terrible', 'bad', 'weak', 'failing', 'worst'],
        'market': ['stock', 'market', 'dow', 'nasdaq', 'economy'],
        'bonds': ['fed', 'interest', 'rates', 'inflation', 'treasury'],
        'gold': ['gold', 'dollar', 'currency']
    }

    found = {}
    for category, keywords in keyword_categories.items():
        matches = [kw for kw in keywords if kw in content_lower]
        found[category] = matches

    return found


def generate_signal(keywords: dict, returns: pd.Series) -> dict:
    """Generate trading signal based on keywords."""
    signal = {
        'SPY': 0, 'QQQ': 0, 'DIA': 0, 'IWM': 0, 'TLT': 0, 'GLD': 0
    }
    reason = []

    # Tariff/Trade keywords -> Long equities
    if keywords.get('bullish'):
        tariff_words = [k for k in keywords['bullish'] if k in ['tariff', 'tariffs', 'trade', 'china']]
        if tariff_words:
            signal['SPY'] = 1
            signal['QQQ'] = 1
            signal['DIA'] = 1
            signal['IWM'] = 1
            reason.append(f"LONG equities ({', '.join(tariff_words)})")

    # Market crash keywords -> Short equities, Long bonds/gold
    if keywords.get('bearish'):
        crash_words = [k for k in keywords['bearish'] if k in ['crash', 'disaster']]
        if crash_words:
            signal['SPY'] = -1
            signal['QQQ'] = -1
            signal['TLT'] = 1  # Flight to safety
            signal['GLD'] = 1  # Flight to safety
            reason.append(f"SHORT equities, LONG safety ({', '.join(crash_words)})")

    # Fed/rates keywords -> Affects bonds
    if keywords.get('bonds'):
        signal['TLT'] = -1 if 'rates' in keywords['bonds'] else 0
        reason.append(f"Bonds affected ({', '.join(keywords['bonds'])})")

    # Gold keywords
    if keywords.get('gold'):
        signal['GLD'] = 1
        reason.append(f"LONG gold ({', '.join(keywords['gold'])})")

    return {'signals': signal, 'reason': '; '.join(reason) if reason else 'No signal'}


def create_report(tweets: pd.DataFrame, all_returns: pd.DataFrame) -> pd.DataFrame:
    """Create the comprehensive report."""
    print("\nCreating report...")

    report = tweets.copy()

    # Add all return columns
    for col in all_returns.columns:
        report[col] = all_returns[col].values

    # Add time features
    report['hour'] = pd.to_datetime(report['entry_time']).dt.hour
    report['day_of_week'] = pd.to_datetime(report['entry_time']).dt.day_name()
    report['market_hours'] = ((report['hour'] >= 9) & (report['hour'] < 16))

    # Detect keywords and generate signals
    keywords_list = []
    signals_list = []

    for idx, row in report.iterrows():
        keywords = detect_keywords(row['tweet_content'])
        keywords_list.append(keywords)

        signal_info = generate_signal(keywords, row)
        signals_list.append(signal_info)

    # Add keyword columns
    report['bullish_keywords'] = [', '.join(k.get('bullish', [])) for k in keywords_list]
    report['bearish_keywords'] = [', '.join(k.get('bearish', [])) for k in keywords_list]
    report['market_keywords'] = [', '.join(k.get('market', [])) for k in keywords_list]

    # Add signal columns
    for ticker in TICKERS:
        report[f'{ticker}_signal'] = [s['signals'][ticker] for s in signals_list]

    report['signal_reason'] = [s['reason'] for s in signals_list]

    # Calculate if signals were correct (5m return)
    for ticker in TICKERS:
        signal_col = f'{ticker}_signal'
        return_col = f'{ticker}_5m'

        if return_col in report.columns:
            report[f'{ticker}_correct'] = (
                ((report[signal_col] == 1) & (report[return_col] > 0)) |
                ((report[signal_col] == -1) & (report[return_col] < 0)) |
                (report[signal_col] == 0)
            )

    return report


def save_csv_report(report: pd.DataFrame, output_path: Path):
    """Save CSV report."""
    # Select columns for CSV
    cols = ['entry_time', 'tweet_content', 'hour', 'day_of_week', 'market_hours',
            'bullish_keywords', 'bearish_keywords', 'market_keywords', 'signal_reason']

    for ticker in TICKERS:
        cols.extend([f'{ticker}_price', f'{ticker}_signal'])
        for mins in HOLDING_PERIODS:
            cols.append(f'{ticker}_{mins}m')

    cols = [c for c in cols if c in report.columns]

    csv_report = report[cols].copy()

    # Format returns as percentages
    for col in csv_report.columns:
        if col.endswith('m') and '_' in col:
            csv_report[col] = csv_report[col].apply(
                lambda x: f'{x*100:.4f}%' if pd.notna(x) and x != 0 else 'N/A'
            )

    csv_report.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  CSV saved: {output_path}")


def save_html_report(report: pd.DataFrame, output_path: Path):
    """Save interactive HTML report."""

    # Calculate summary stats
    total_tweets = len(report)
    tweets_with_signal = (report[[f'{t}_signal' for t in TICKERS]].abs().sum(axis=1) > 0).sum()

    # Signal accuracy by ticker
    accuracy_stats = {}
    for ticker in TICKERS:
        signal_col = f'{ticker}_signal'
        correct_col = f'{ticker}_correct'
        if correct_col in report.columns:
            has_signal = report[signal_col] != 0
            if has_signal.sum() > 0:
                accuracy_stats[ticker] = report.loc[has_signal, correct_col].mean() * 100

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Multi-Asset Tweet Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f7fa; }}
        h1 {{ color: #1a1a2e; }}
        .summary {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-top: 15px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 20px; font-weight: bold; }}
        .stat-label {{ font-size: 11px; color: #666; margin-top: 5px; }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        .neutral {{ color: #6b7280; }}
        .filters {{ background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .filter-btn {{ padding: 6px 12px; margin-right: 5px; border: none; border-radius: 5px; cursor: pointer; font-size: 12px; }}
        .filter-btn.active {{ background: #3b82f6; color: white; }}
        .filter-btn:not(.active) {{ background: #e5e7eb; }}
        input[type="text"] {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 5px; width: 250px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; font-size: 11px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th {{ background: #1e3a5f; color: white; padding: 10px 6px; text-align: left; position: sticky; top: 0; }}
        td {{ padding: 8px 6px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8fafc; }}
        .tweet-content {{ max-width: 300px; word-wrap: break-word; font-size: 10px; }}
        .signal-long {{ background: #d1fae5; color: #065f46; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; }}
        .signal-short {{ background: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; }}
        .signal-none {{ color: #9ca3af; font-size: 10px; }}
        .return-pos {{ color: #10b981; }}
        .return-neg {{ color: #ef4444; }}
        .ticker-header {{ background: #2d4a6f !important; }}
        .keywords {{ font-size: 9px; color: #666; max-width: 100px; }}
    </style>
</head>
<body>
    <h1>Multi-Asset Tweet Signal Report</h1>

    <div class="summary">
        <h3>Summary Statistics</h3>
        <p>Total Tweets: <strong>{total_tweets:,}</strong> | Tweets with Signals: <strong>{tweets_with_signal:,}</strong></p>
        <div class="stats-grid">
"""

    for ticker in TICKERS:
        acc = accuracy_stats.get(ticker, 0)
        color = 'positive' if acc > 50 else 'negative' if acc < 50 else 'neutral'
        html_content += f"""
            <div class="stat-card">
                <div class="stat-value {color}">{acc:.1f}%</div>
                <div class="stat-label">{ticker} Signal Accuracy</div>
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="filters">
        <input type="text" id="search" placeholder="Search tweets..." onkeyup="filterTable()">
        <button class="filter-btn active" onclick="filterBySignal('all')">All</button>
        <button class="filter-btn" onclick="filterBySignal('signals')">With Signals</button>
        <button class="filter-btn" onclick="filterBySignal('long')">Long Signals</button>
        <button class="filter-btn" onclick="filterBySignal('short')">Short Signals</button>
    </div>

    <div style="overflow-x: auto;">
    <table id="reportTable">
        <thead>
            <tr>
                <th>Time</th>
                <th>Tweet</th>
                <th>Keywords</th>
"""

    for ticker in TICKERS:
        html_content += f'<th class="ticker-header">{ticker}</th>'

    html_content += """
                <th>Signal Reason</th>
            </tr>
        </thead>
        <tbody>
"""

    # Add rows (most recent first, limit to 5000 for performance)
    report_sorted = report.sort_values('entry_time', ascending=False).head(5000)

    for idx, row in report_sorted.iterrows():
        # Determine if row has any signal
        has_long = any(row.get(f'{t}_signal', 0) == 1 for t in TICKERS)
        has_short = any(row.get(f'{t}_signal', 0) == -1 for t in TICKERS)
        signal_type = 'long' if has_long else ('short' if has_short else 'none')

        time_str = pd.to_datetime(row['entry_time']).strftime('%Y-%m-%d %H:%M')
        content = str(row['tweet_content'])[:150] + '...' if pd.notna(row['tweet_content']) and len(str(row['tweet_content'])) > 150 else (row['tweet_content'] if pd.notna(row['tweet_content']) else '')

        # Keywords
        kw_parts = []
        if row.get('bullish_keywords'): kw_parts.append(f"Bull: {row['bullish_keywords']}")
        if row.get('bearish_keywords'): kw_parts.append(f"Bear: {row['bearish_keywords']}")
        keywords_str = '<br>'.join(kw_parts) if kw_parts else '-'

        html_content += f'<tr data-signal="{signal_type}">'
        html_content += f'<td>{time_str}</td>'
        html_content += f'<td class="tweet-content">{content}</td>'
        html_content += f'<td class="keywords">{keywords_str}</td>'

        # Each ticker column
        for ticker in TICKERS:
            signal = row.get(f'{ticker}_signal', 0)
            ret_5m = row.get(f'{ticker}_5m', 0)
            ret_15m = row.get(f'{ticker}_15m', 0)
            price = row.get(f'{ticker}_price', 0)

            if signal == 1:
                signal_str = '<span class="signal-long">LONG</span>'
            elif signal == -1:
                signal_str = '<span class="signal-short">SHORT</span>'
            else:
                signal_str = '<span class="signal-none">-</span>'

            ret_5m_str = f'{ret_5m*100:+.3f}%' if pd.notna(ret_5m) and ret_5m != 0 else 'N/A'
            ret_15m_str = f'{ret_15m*100:+.3f}%' if pd.notna(ret_15m) and ret_15m != 0 else 'N/A'

            ret_class_5 = 'return-pos' if ret_5m > 0 else 'return-neg' if ret_5m < 0 else ''
            ret_class_15 = 'return-pos' if ret_15m > 0 else 'return-neg' if ret_15m < 0 else ''

            price_str = f'${price:.2f}' if pd.notna(price) and price > 0 else ''

            html_content += f'''<td>
                {signal_str}<br>
                <span style="font-size:9px;color:#888">{price_str}</span><br>
                <span class="{ret_class_5}" style="font-size:10px">5m: {ret_5m_str}</span><br>
                <span class="{ret_class_15}" style="font-size:10px">15m: {ret_15m_str}</span>
            </td>'''

        reason = row.get('signal_reason', '')
        html_content += f'<td style="font-size:10px">{reason}</td>'
        html_content += '</tr>'

    html_content += """
        </tbody>
    </table>
    </div>

    <script>
        function filterTable() {
            const input = document.getElementById('search').value.toLowerCase();
            const rows = document.querySelectorAll('#reportTable tbody tr');
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(input) ? '' : 'none';
            });
        }

        function filterBySignal(type) {
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            const rows = document.querySelectorAll('#reportTable tbody tr');
            rows.forEach(row => {
                const signal = row.getAttribute('data-signal');
                if (type === 'all') {
                    row.style.display = '';
                } else if (type === 'signals') {
                    row.style.display = (signal === 'long' || signal === 'short') ? '' : 'none';
                } else {
                    row.style.display = signal === type ? '' : 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  HTML saved: {output_path}")


def main():
    print("="*70)
    print("MULTI-ASSET TWEET SIGNAL REPORT")
    print("="*70)
    print(f"Assets: {', '.join(TICKERS)}")
    print(f"Holding periods: {HOLDING_PERIODS} minutes")

    # Load tweets
    tweets = load_tweet_data()

    # Calculate returns for each ticker
    print("\nCalculating returns for each asset...")
    all_returns = pd.DataFrame(index=tweets.index)

    for ticker in TICKERS:
        print(f"  Processing {ticker}...")
        market_pl = load_market_data(ticker)

        if market_pl is not None:
            ticker_returns = calculate_returns_for_ticker(tweets, market_pl, ticker)
            for col in ticker_returns.columns:
                all_returns[col] = ticker_returns[col].values

    # Create report
    report = create_report(tweets, all_returns)

    # Save reports
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print("\nSaving reports...")
    save_csv_report(report, output_dir / 'multi_asset_tweet_report.csv')
    save_html_report(report, output_dir / 'multi_asset_tweet_report.html')

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total tweets: {len(report):,}")

    for ticker in TICKERS:
        signal_col = f'{ticker}_signal'
        if signal_col in report.columns:
            long_count = (report[signal_col] == 1).sum()
            short_count = (report[signal_col] == -1).sum()
            print(f"  {ticker}: {long_count} LONG, {short_count} SHORT signals")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print("\nOutput files:")
    print("  output/multi_asset_tweet_report.csv")
    print("  output/multi_asset_tweet_report.html")
    print("\nOpen the HTML file in a browser for interactive viewing.")


if __name__ == "__main__":
    main()
