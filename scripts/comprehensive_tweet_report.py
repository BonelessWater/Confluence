"""
COMPREHENSIVE TWEET-SIGNAL REPORT

Generates a detailed report of ALL tweets with their trading signals
for manual verification. Includes:
- Full tweet content
- Timestamp
- Market conditions
- Model predictions
- Keywords detected
- Forward returns at multiple timeframes
- Signal correctness
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


def load_all_data(ticker='SPY'):
    """Load features and market data."""
    print(f"\n{'='*60}")
    print(f"Loading data for {ticker}")
    print(f"{'='*60}")

    features_path = Path(__file__).parent.parent / 'data' / 'trump-truth-social-archive' / 'data' / 'truth_archive_with_features.parquet'
    df = pd.read_parquet(features_path)
    df = df[df['ticker'] == ticker].copy()

    market_path = Path(__file__).parent.parent / 'data' / f'{ticker}.parquet'
    market_pl = pl.read_parquet(market_path)

    print(f"  Total tweets: {len(df)}")

    return df, market_pl


def calculate_all_returns(df, market_pl):
    """Calculate returns at multiple timeframes."""
    print("\nCalculating returns at multiple timeframes...")

    if df['entry_time'].dt.tz is not None:
        df['entry_time'] = df['entry_time'].dt.tz_localize(None)

    min_time = df['entry_time'].min()
    max_time = df['entry_time'].max() + pd.Timedelta(hours=2)

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

    entry_times = df['entry_time'].dt.floor('min')
    df['entry_price'] = price_df['close'].reindex(entry_times).values

    # Calculate returns for multiple holding periods
    for mins in [1, 5, 15, 30, 60]:
        exit_times = (df['entry_time'] + pd.Timedelta(minutes=mins)).dt.floor('min')
        exit_prices = price_df['close'].reindex(exit_times).values
        df[f'return_{mins}m'] = (exit_prices - df['entry_price']) / df['entry_price']
        df[f'return_{mins}m'] = df[f'return_{mins}m'].fillna(0)

    # Past returns for context
    for mins in [5, 15, 30]:
        past_times = (df['entry_time'] - pd.Timedelta(minutes=mins)).dt.floor('min')
        past_prices = price_df['close'].reindex(past_times).values
        df[f'past_{mins}m'] = (df['entry_price'] - past_prices) / past_prices
        df[f'past_{mins}m'] = df[f'past_{mins}m'].fillna(0)

    valid = (df['return_5m'] != 0).sum()
    print(f"  Valid returns: {valid}")

    return df


def detect_keywords(df):
    """Detect keywords in tweet content."""
    print("\nDetecting keywords...")

    # Keyword categories
    bullish = ['tariff', 'tariffs', 'trade', 'china', 'jobs', 'growth', 'strong', 'great', 'winning', 'record', 'best', 'beautiful', 'amazing']
    bearish = ['crash', 'disaster', 'terrible', 'bad', 'weak', 'failing', 'sad', 'worst', 'corrupt', 'fake']
    market = ['stock', 'market', 'economy', 'dow', 'nasdaq', 's&p']
    political = ['democrat', 'republican', 'biden', 'pelosi', 'congress', 'senate']

    df['content_lower'] = df['tweet_content'].fillna('').str.lower()

    def get_keywords(text, keyword_list):
        found = [kw for kw in keyword_list if kw in text]
        return ', '.join(found) if found else ''

    df['bullish_keywords'] = df['content_lower'].apply(lambda x: get_keywords(x, bullish))
    df['bearish_keywords'] = df['content_lower'].apply(lambda x: get_keywords(x, bearish))
    df['market_keywords'] = df['content_lower'].apply(lambda x: get_keywords(x, market))
    df['political_keywords'] = df['content_lower'].apply(lambda x: get_keywords(x, political))

    # Count keywords
    df['n_bullish'] = df['bullish_keywords'].apply(lambda x: len(x.split(', ')) if x else 0)
    df['n_bearish'] = df['bearish_keywords'].apply(lambda x: len(x.split(', ')) if x else 0)

    print(f"  Tweets with bullish keywords: {(df['n_bullish'] > 0).sum()}")
    print(f"  Tweets with bearish keywords: {(df['n_bearish'] > 0).sum()}")
    print(f"  Tweets with market keywords: {(df['market_keywords'] != '').sum()}")

    return df


def generate_signals(df):
    """Generate trading signals based on keywords and market conditions."""
    print("\nGenerating signals...")

    df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
    df['market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)

    # Simple keyword-based signal
    df['signal'] = 'HOLD'

    # Long on tariff/china (strongest positive keywords)
    strong_long = df['content_lower'].str.contains('tariff|china|trade deal', na=False, regex=True)
    df.loc[strong_long & (df['market_hours'] == 1) & (df['return_5m'] != 0), 'signal'] = 'LONG'

    # Short on market crash mentions
    strong_short = df['content_lower'].str.contains('market crash|stock crash|crash', na=False, regex=True)
    df.loc[strong_short & (df['market_hours'] == 1) & (df['return_5m'] != 0), 'signal'] = 'SHORT'

    print(f"  LONG signals: {(df['signal'] == 'LONG').sum()}")
    print(f"  SHORT signals: {(df['signal'] == 'SHORT').sum()}")
    print(f"  HOLD: {(df['signal'] == 'HOLD').sum()}")

    return df


def calculate_signal_accuracy(df):
    """Calculate if signals were correct."""
    df['signal_correct'] = False

    # LONG is correct if 5m return is positive
    df.loc[(df['signal'] == 'LONG') & (df['return_5m'] > 0), 'signal_correct'] = True

    # SHORT is correct if 5m return is negative
    df.loc[(df['signal'] == 'SHORT') & (df['return_5m'] < 0), 'signal_correct'] = True

    # HOLD is always "correct" (neutral)
    df.loc[df['signal'] == 'HOLD', 'signal_correct'] = True

    return df


def generate_reports(df, ticker, output_dir='output'):
    """Generate CSV and HTML reports."""
    print(f"\n{'='*60}")
    print("GENERATING REPORTS")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Prepare report columns
    report_df = df[[
        'entry_time', 'tweet_content', 'signal',
        'bullish_keywords', 'bearish_keywords', 'market_keywords',
        'return_1m', 'return_5m', 'return_15m', 'return_30m', 'return_60m',
        'past_5m', 'past_15m', 'past_30m',
        'entry_price', 'signal_correct', 'hour', 'market_hours'
    ]].copy()

    report_df = report_df.sort_values('entry_time', ascending=False)

    # Format returns as percentages
    for col in ['return_1m', 'return_5m', 'return_15m', 'return_30m', 'return_60m',
                'past_5m', 'past_15m', 'past_30m']:
        report_df[col] = report_df[col].apply(lambda x: f'{x*100:+.3f}%' if pd.notna(x) and x != 0 else 'N/A')

    report_df['entry_price'] = report_df['entry_price'].apply(lambda x: f'${x:.2f}' if pd.notna(x) else 'N/A')
    report_df['entry_time'] = pd.to_datetime(report_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save CSV
    csv_path = output_path / f'{ticker}_comprehensive_tweet_report.csv'
    report_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  CSV saved: {csv_path}")

    # Calculate statistics
    signals_only = df[df['signal'] != 'HOLD'].copy()
    long_signals = signals_only[signals_only['signal'] == 'LONG']
    short_signals = signals_only[signals_only['signal'] == 'SHORT']

    total = len(df)
    n_long = len(long_signals)
    n_short = len(short_signals)
    n_hold = total - n_long - n_short

    long_accuracy = long_signals['signal_correct'].mean() * 100 if n_long > 0 else 0
    short_accuracy = short_signals['signal_correct'].mean() * 100 if n_short > 0 else 0
    avg_long_ret = long_signals['return_5m'].mean() * 100 if n_long > 0 else 0
    avg_short_ret = short_signals['return_5m'].mean() * 100 if n_short > 0 else 0

    # Generate HTML
    html_path = output_path / f'{ticker}_comprehensive_tweet_report.html'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Tweet Signal Report - {ticker}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f0f2f5; }}
        h1 {{ color: #1a1a2e; margin-bottom: 20px; }}
        .stats-container {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        .neutral {{ color: #6b7280; }}
        .filters {{ background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .filter-btn {{ padding: 8px 16px; margin-right: 8px; border: none; border-radius: 6px; cursor: pointer; font-weight: 500; }}
        .filter-btn.active {{ background: #3b82f6; color: white; }}
        .filter-btn:not(.active) {{ background: #e5e7eb; color: #374151; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th {{ background: #1e3a5f; color: white; padding: 12px 10px; text-align: left; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
        td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; font-size: 12px; }}
        tr:hover {{ background: #f8fafc; }}
        .tweet-content {{ max-width: 350px; word-wrap: break-word; line-height: 1.4; }}
        .signal-long {{ background: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 15px; font-weight: 600; font-size: 11px; }}
        .signal-short {{ background: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 15px; font-weight: 600; font-size: 11px; }}
        .signal-hold {{ background: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 15px; font-weight: 600; font-size: 11px; }}
        .correct {{ color: #10b981; font-weight: bold; }}
        .incorrect {{ color: #ef4444; font-weight: bold; }}
        .keywords {{ font-size: 10px; color: #6b7280; }}
        .return-pos {{ color: #10b981; font-weight: 500; }}
        .return-neg {{ color: #ef4444; font-weight: 500; }}
        .search-box {{ padding: 10px 15px; border: 1px solid #d1d5db; border-radius: 8px; width: 300px; margin-right: 15px; }}
    </style>
</head>
<body>
    <h1>Comprehensive Tweet Signal Report - {ticker}</h1>

    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">{total:,}</div>
            <div class="stat-label">Total Tweets</div>
        </div>
        <div class="stat-card">
            <div class="stat-value positive">{n_long}</div>
            <div class="stat-label">LONG Signals ({long_accuracy:.1f}% accurate)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value negative">{n_short}</div>
            <div class="stat-label">SHORT Signals ({short_accuracy:.1f}% accurate)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value neutral">{n_hold}</div>
            <div class="stat-label">HOLD (No Signal)</div>
        </div>
    </div>

    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value {'positive' if avg_long_ret > 0 else 'negative'}">{avg_long_ret:+.4f}%</div>
            <div class="stat-label">Avg LONG Return (5m)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {'positive' if avg_short_ret > 0 else 'negative'}">{avg_short_ret:+.4f}%</div>
            <div class="stat-label">Avg SHORT Return (5m)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{(df['return_5m'] != 0).sum():,}</div>
            <div class="stat-label">Valid Market Data</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{df['market_hours'].sum():,}</div>
            <div class="stat-label">During Market Hours</div>
        </div>
    </div>

    <div class="filters">
        <input type="text" class="search-box" id="searchBox" placeholder="Search tweets..." onkeyup="searchTable()">
        <button class="filter-btn active" onclick="filterTable('all')">All</button>
        <button class="filter-btn" onclick="filterTable('LONG')">LONG Only</button>
        <button class="filter-btn" onclick="filterTable('SHORT')">SHORT Only</button>
        <button class="filter-btn" onclick="filterTable('signals')">All Signals</button>
        <button class="filter-btn" onclick="filterTable('correct')">Correct Only</button>
        <button class="filter-btn" onclick="filterTable('incorrect')">Incorrect Only</button>
    </div>

    <table id="tweetTable">
        <thead>
            <tr>
                <th>Time</th>
                <th>Tweet Content</th>
                <th>Signal</th>
                <th>Keywords</th>
                <th>1m</th>
                <th>5m</th>
                <th>15m</th>
                <th>30m</th>
                <th>Correct?</th>
            </tr>
        </thead>
        <tbody>
"""

    # Add rows
    for idx, row in report_df.iterrows():
        signal_class = f"signal-{row['signal'].lower()}"
        content = str(row['tweet_content'])[:200] + '...' if pd.notna(row['tweet_content']) and len(str(row['tweet_content'])) > 200 else (row['tweet_content'] if pd.notna(row['tweet_content']) else 'N/A')

        # Combine keywords
        keywords = []
        if row['bullish_keywords']: keywords.append(f"Bull: {row['bullish_keywords']}")
        if row['bearish_keywords']: keywords.append(f"Bear: {row['bearish_keywords']}")
        if row['market_keywords']: keywords.append(f"Mkt: {row['market_keywords']}")
        keywords_str = '<br>'.join(keywords) if keywords else '-'

        # Format returns with colors
        def format_return(r):
            if r == 'N/A': return r
            try:
                val = float(r.replace('%', ''))
                cls = 'return-pos' if val > 0 else 'return-neg' if val < 0 else ''
                return f'<span class="{cls}">{r}</span>'
            except:
                return r

        correct_class = 'correct' if row['signal_correct'] else 'incorrect'
        correct_text = 'Yes' if row['signal_correct'] else 'No'
        if row['signal'] == 'HOLD':
            correct_class = ''
            correct_text = '-'

        html_content += f"""
            <tr data-signal="{row['signal']}" data-correct="{row['signal_correct']}">
                <td>{row['entry_time']}</td>
                <td class="tweet-content">{content}</td>
                <td><span class="{signal_class}">{row['signal']}</span></td>
                <td class="keywords">{keywords_str}</td>
                <td>{format_return(row['return_1m'])}</td>
                <td>{format_return(row['return_5m'])}</td>
                <td>{format_return(row['return_15m'])}</td>
                <td>{format_return(row['return_30m'])}</td>
                <td class="{correct_class}">{correct_text}</td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>

    <script>
        function filterTable(filter) {
            const rows = document.querySelectorAll('#tweetTable tbody tr');
            const buttons = document.querySelectorAll('.filter-btn');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            rows.forEach(row => {
                const signal = row.getAttribute('data-signal');
                const correct = row.getAttribute('data-correct');

                let show = true;
                if (filter === 'LONG') show = signal === 'LONG';
                else if (filter === 'SHORT') show = signal === 'SHORT';
                else if (filter === 'signals') show = signal !== 'HOLD';
                else if (filter === 'correct') show = correct === 'True' && signal !== 'HOLD';
                else if (filter === 'incorrect') show = correct === 'False' && signal !== 'HOLD';

                row.style.display = show ? '' : 'none';
            });
        }

        function searchTable() {
            const input = document.getElementById('searchBox').value.toLowerCase();
            const rows = document.querySelectorAll('#tweetTable tbody tr');

            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(input) ? '' : 'none';
            });
        }
    </script>
</body>
</html>
"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  HTML saved: {html_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tweets: {total:,}")
    print(f"  LONG signals: {n_long} (accuracy: {long_accuracy:.1f}%)")
    print(f"  SHORT signals: {n_short} (accuracy: {short_accuracy:.1f}%)")
    print(f"  HOLD: {n_hold}")
    print(f"\n  Average LONG return (5m): {avg_long_ret:+.4f}%")
    print(f"  Average SHORT return (5m): {avg_short_ret:+.4f}%")

    return report_df


def main():
    print("="*70)
    print("COMPREHENSIVE TWEET-SIGNAL REPORT GENERATOR")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ticker = 'SPY'

    # Load data
    df, market_pl = load_all_data(ticker)

    # Calculate returns
    df = calculate_all_returns(df, market_pl)

    # Detect keywords
    df = detect_keywords(df)

    # Generate signals
    df = generate_signals(df)

    # Calculate accuracy
    df = calculate_signal_accuracy(df)

    # Generate reports
    report_df = generate_reports(df, ticker)

    print(f"\n{'='*70}")
    print("[OK] REPORT GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFiles created:")
    print(f"  1. output/{ticker}_comprehensive_tweet_report.csv")
    print(f"  2. output/{ticker}_comprehensive_tweet_report.html")
    print(f"\nOpen the HTML file in a browser for interactive filtering and search.")


if __name__ == "__main__":
    main()
