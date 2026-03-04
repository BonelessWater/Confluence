"""
Strategy Report: Sentiment & Correlation Strategy Analysis

Generates:
  1. Per-method text report  (signal quality, GOOD/SUSPECT labeling)
  2. Full trade log CSV      (tweet → signal → position → hold time → outcome)
  3. Quantstats HTML tearsheet (vs SPY benchmark)
  4. Side-by-side comparison report

Usage:
    python scripts/strategy_report.py [--method sentiment|correlation|both]
    python scripts/strategy_report.py --output output/strategy_report
"""

import sys
import os
import argparse
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import polars as pl

from config.settings import (
    FEATURES_PARQUET, DATA_DIR, OUTPUT_DIR,
    INITIAL_CAPITAL, HOLDING_PERIOD_MINUTES,
)
from src.tweet_ticker_analysis.tweet_cleaner import TweetCleaner
from src.tweet_ticker_analysis.correlation_discovery import CorrelationDiscovery
from src.tweet_ticker_analysis.sentiment_analyzer import FinancialSentimentAnalyzer, SentimentScorer
from src.tweet_ticker_analysis.backtest_tweet_strategies import TweetStrategyBacktester
from src.tweet_ticker_analysis.ticker_relevance_filter import TickerRelevanceFilter

try:
    import quantstats as qs
    QS_AVAILABLE = True
except ImportError:
    QS_AVAILABLE = False
    print("Note: quantstats not installed — HTML tearsheets will be skipped.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Price loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_price_df(price_path: Path) -> Optional[pd.DataFrame]:
    try:
        pdf = pl.read_parquet(price_path).to_pandas()
        if 'ts_event' in pdf.columns:
            pdf['timestamp'] = pd.to_datetime(pdf['ts_event'])
        elif 'timestamp' in pdf.columns:
            pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
        elif isinstance(pdf.index, pd.DatetimeIndex):
            pdf = pdf.copy()
            pdf['timestamp'] = pdf.index
        else:
            return None
        if pdf['timestamp'].dt.tz is not None:
            pdf['timestamp'] = pdf['timestamp'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
        pdf = pdf.set_index('timestamp').sort_index()
        return pdf if 'close' in pdf.columns else None
    except Exception as e:
        print(f"    Warning loading {price_path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(min_relevance: float = 0.15) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, List[str]]:
    features_path = FEATURES_PARQUET
    if not features_path.exists():
        alt = DATA_DIR / "trump-truth-social-archive" / "data" / "truth_archive_with_embeddings.parquet"
        if alt.exists():
            features_path = alt
        else:
            raise FileNotFoundError(f"Tweet features not found: {features_path}")

    print(f"Loading tweets from {features_path}...")
    tweets_df = pd.read_parquet(features_path)
    tweets_df['entry_time'] = pd.to_datetime(tweets_df['entry_time'])
    if tweets_df['entry_time'].dt.tz is not None:
        tweets_df['entry_time'] = (
            tweets_df['entry_time'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
        )
    print(f"  Raw rows: {len(tweets_df):,}  unique tweets: {tweets_df['tweet_id'].nunique():,}")

    cleaner = TweetCleaner()
    tweets_df = cleaner.clean_tweets(tweets_df, text_column='tweet_content')

    print(f"\nApplying relevance filter (min_score={min_relevance})...")
    rf = TickerRelevanceFilter(min_relevance_score=min_relevance)
    before = len(tweets_df)
    tweets_df = rf.filter_all_tickers(tweets_df)
    print(f"  {before:,} → {len(tweets_df):,} rows  ({len(tweets_df)/max(1,before)*100:.1f}% kept)")
    print(f"  Unique tweets: {tweets_df['tweet_id'].nunique():,}")

    print("\nLoading price data...")
    price_data: Dict[str, pd.DataFrame] = {}
    for pfile in sorted(DATA_DIR.glob('*.parquet')):
        ticker = pfile.stem
        pdf = _load_price_df(pfile)
        if pdf is not None:
            price_data[ticker] = pdf
            print(f"  {ticker}: {len(pdf):,} bars  [{pdf.index.min()} → {pdf.index.max()}]")
        else:
            print(f"  {ticker}: skipped (no 'close' column)")

    available = sorted(set(tweets_df['ticker'].unique()) & set(price_data.keys()))
    print(f"\nTickers with tweet+price data: {available}")
    tweets_df = tweets_df[tweets_df['ticker'].isin(available)].copy()

    print("\nCalculating forward returns...")
    returns_df = _calc_returns(tweets_df, price_data, available)

    return tweets_df, price_data, returns_df, available


def _calc_returns(tweets_df: pd.DataFrame, price_data: Dict, tickers: List[str]) -> pd.DataFrame:
    ts = tweets_df.sort_values('entry_time').copy()
    horizons = [5, 15, 30, 60]
    all_ret = []
    for ticker in tickers:
        if ticker not in price_data:
            continue
        pdf = price_data[ticker].sort_index()
        if 'close' not in pdf.columns:
            continue
        tw = ts[ts['ticker'] == ticker][['tweet_id', 'entry_time']].copy()
        if tw.empty:
            continue
        tmp = pd.merge_asof(tw, pdf[['close']], left_on='entry_time', right_index=True, direction='forward')
        tmp.rename(columns={'close': 'entry_price'}, inplace=True)
        for h in horizons:
            tw[f'exit_time_{h}m'] = tw['entry_time'] + pd.Timedelta(minutes=h)
            ex = pd.merge_asof(tw[[f'exit_time_{h}m']], pdf[['close']],
                               left_on=f'exit_time_{h}m', right_index=True, direction='forward')
            tmp[f'{ticker}_{h}m'] = ((ex['close'] - tmp['entry_price']) / tmp['entry_price']).fillna(0.0)
        all_ret.append(tmp.drop(columns=['entry_time', 'entry_price']))
    return pd.concat(all_ret, ignore_index=True) if all_ret else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Trade auditing — full detail per trade
# ─────────────────────────────────────────────────────────────────────────────

def _trade_sense(relevance: float, influence: float) -> str:
    if relevance >= 0.4 and abs(influence) > 0.0002:
        return 'GOOD'
    elif relevance >= 0.15:
        return 'MARGINAL'
    return 'SUSPECT'


def build_tweet_map(tweets_df: pd.DataFrame) -> Dict[str, str]:
    """Return {tweet_id_str: full_text} and also an index by content prefix."""
    m: Dict[str, str] = {}
    if 'tweet_id' in tweets_df.columns and 'tweet_content' in tweets_df.columns:
        for _, row in tweets_df.drop_duplicates('tweet_id').iterrows():
            m[str(row['tweet_id'])] = str(row['tweet_content'])
    return m


def resolve_full_text(snippet: str, tweet_map: Dict[str, str]) -> str:
    """Best-effort: match snippet to full tweet text."""
    if not snippet:
        return ''
    prefix = snippet[:80]
    for txt in tweet_map.values():
        if txt.startswith(prefix):
            return txt
    return snippet


def audit_trades(trades_df: pd.DataFrame,
                 tweets_df: pd.DataFrame,
                 rf: TickerRelevanceFilter) -> pd.DataFrame:
    """
    Produce a full enriched trade log with columns:
      entry_time, exit_time, hold_minutes, ticker, direction,
      position_size_usd, weight_pct, influence_score,
      gross_return_pct, net_return_pct, transaction_cost_usd, pnl_usd,
      capital_after, relevance_score, signal_sense,
      signal_reason, tweet_text
    """
    if trades_df.empty:
        return pd.DataFrame()

    tweet_map = build_tweet_map(tweets_df)

    rows = []
    for _, t in trades_df.iterrows():
        snippet = str(t.get('tweet_snippet', ''))
        ticker  = str(t.get('ticker', ''))
        full    = resolve_full_text(snippet, tweet_map)
        rel     = rf.score_relevance(full, ticker)
        inf     = float(t.get('influence_score', 0.0))

        rows.append({
            'entry_time':          t.get('entry_time'),
            'exit_time':           t.get('exit_time'),
            'hold_minutes':        round(float(t.get('duration_minutes', 0.0)), 1),
            'ticker':              ticker,
            'direction':           t.get('direction', '?'),
            'position_size_usd':   round(float(t.get('position_size_usd', 0.0)), 2),
            'weight_pct':          round(float(t.get('weight', 0.0)) * 100, 3),
            'influence_score':     round(inf, 7),
            'gross_return_pct':    round(float(t.get('gross_return', 0.0)) * 100, 5),
            'net_return_pct':      round(float(t.get('net_return', 0.0)) * 100, 5),
            'transaction_cost_usd': round(float(t.get('transaction_cost', 0.0)), 4),
            'pnl_usd':             round(float(t.get('pnl', 0.0)), 2),
            'capital_after':       round(float(t.get('capital', INITIAL_CAPITAL)), 2),
            'relevance_score':     round(rel, 3),
            'signal_sense':        _trade_sense(rel, inf),
            'signal_reason':       str(t.get('signal_reason', '')),
            'tweet_text':          full,
        })

    return pd.DataFrame(rows).sort_values('entry_time').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Quantstats report
# ─────────────────────────────────────────────────────────────────────────────

def generate_quantstats(method_name: str,
                        equity_df: pd.DataFrame,
                        price_data: Dict,
                        output_dir: Path):
    """Generate quantstats HTML tearsheet + basic matplotlib equity plot."""
    if equity_df.empty:
        print(f"  Skipping quantstats for {method_name} — empty equity curve.")
        return

    eq = equity_df.copy()
    eq['time'] = pd.to_datetime(eq['time'])
    eq = eq.set_index('time').sort_index()

    # Resample to per-minute (fill forward) then to daily for quantstats
    eq_min = eq['equity'].resample('1min').last().ffill()
    daily  = eq_min.resample('D').last().ffill().dropna()
    rets   = daily.pct_change().dropna()

    if len(rets) < 3:
        print(f"  Not enough daily observations for {method_name} quantstats.")
        return

    # ── SPY benchmark ──
    bench = None
    if 'SPY' in price_data:
        spy_daily = price_data['SPY']['close'].resample('D').last().ffill().dropna()
        common = rets.index.intersection(spy_daily.index)
        if len(common) >= 3:
            bench = spy_daily.pct_change().dropna().reindex(common).ffill().dropna()
            rets  = rets.reindex(common).ffill().dropna()
            bench = bench.reindex(rets.index).dropna()

    # ── HTML tearsheet ──
    if QS_AVAILABLE:
        try:
            html_path = output_dir / f'{method_name}_quantstats.html'
            if bench is not None and len(bench) >= 3:
                qs.reports.html(
                    rets, benchmark=bench,
                    output=str(html_path),
                    title=f'{method_name.upper()} Strategy vs SPY',
                    compounded=True,
                )
            else:
                qs.reports.html(
                    rets,
                    output=str(html_path),
                    title=f'{method_name.upper()} Strategy',
                    compounded=True,
                )
            print(f"  Quantstats tearsheet → {html_path}")
        except Exception as e:
            print(f"  Quantstats HTML failed: {e}")

    # ── Matplotlib charts ──
    if MPL_AVAILABLE:
        _plot_equity_and_drawdown(method_name, eq_min, rets, bench, output_dir)


def _plot_equity_and_drawdown(method_name: str,
                               eq_min: pd.Series,
                               rets: pd.Series,
                               bench: Optional[pd.Series],
                               output_dir: Path):
    try:
        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(3, 1, hspace=0.45)

        # ── Equity curve ──
        ax1 = fig.add_subplot(gs[0])
        norm = eq_min / eq_min.iloc[0] * 100 - 100
        ax1.plot(norm.index, norm.values, color='steelblue', linewidth=1.5, label=method_name)
        if bench is not None:
            bench_cum = (1 + bench).cumprod()
            bench_norm = bench_cum / bench_cum.iloc[0] * 100 - 100
            ax1.plot(bench_norm.index, bench_norm.values, color='grey',
                     linewidth=1, linestyle='--', alpha=0.8, label='SPY')
        ax1.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax1.set_title(f'{method_name.upper()} — Equity Curve (% return from start)',
                      fontsize=11, fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ── Drawdown ──
        ax2 = fig.add_subplot(gs[1])
        cumret = (1 + rets).cumprod()
        rolling_max = cumret.cummax()
        drawdown = (cumret - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='crimson', alpha=0.4)
        ax2.plot(drawdown.index, drawdown.values, color='crimson', linewidth=0.8)
        ax2.set_title('Daily Drawdown (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # ── Monthly returns bar ──
        ax3 = fig.add_subplot(gs[2])
        monthly = rets.resample('ME').apply(lambda r: (1 + r).prod() - 1) * 100
        colors  = ['steelblue' if v >= 0 else 'crimson' for v in monthly.values]
        ax3.bar(monthly.index, monthly.values, color=colors, width=20)
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_title('Monthly Returns (%)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'{method_name.upper()} Strategy Performance', fontsize=13, y=0.98)
        out = output_dir / f'{method_name}_performance.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Performance chart  → {out}")
    except Exception as e:
        print(f"  Chart generation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Text report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sec(title: str, char: str = '=', w: int = 100) -> str:
    return f"\n{'=' * w}\n{title}\n{'=' * w}" if char == '=' else f"\n{'-' * w}\n{title}\n{'-' * w}"


def _wrap(text: str, width: int = 96, indent: int = 4) -> str:
    return textwrap.fill(str(text), width=width,
                         initial_indent=' ' * indent, subsequent_indent=' ' * indent)


def _fmt_trade_block(row: pd.Series) -> List[str]:
    """Format one audited trade as a readable block."""
    lines = []
    sense_tag = f"  *** {row['signal_sense']} ***" if row['signal_sense'] != 'MARGINAL' else ''
    pnl_sign  = '+' if row['pnl_usd'] >= 0 else ''
    lines.append(
        f"\n  ┌─ [{row['entry_time']}] → [{row['exit_time']}]"
        f"  hold={row['hold_minutes']:.0f} min{sense_tag}"
    )
    lines.append(
        f"  │  {row['ticker']:5s} {row['direction']:5s} | "
        f"pos=${row['position_size_usd']:>10,.2f}  wt={row['weight_pct']:.3f}% | "
        f"influence={row['influence_score']:+.6f}"
    )
    lines.append(
        f"  │  gross={row['gross_return_pct']:+.4f}%  net={row['net_return_pct']:+.4f}%  "
        f"txcost=${row['transaction_cost_usd']:.4f}  "
        f"P&L={pnl_sign}${row['pnl_usd']:.2f}  capital=${row['capital_after']:,.2f}"
    )
    lines.append(
        f"  │  relevance={row['relevance_score']:.3f}  "
        f"signal: {str(row['signal_reason'])[:70]}"
    )
    tweet = str(row['tweet_text'])
    lines.append(f"  │  Tweet:")
    # Wrap at 90 chars, indented under the box
    for wl in textwrap.wrap(tweet, width=88):
        lines.append(f"  │    {wl}")
    lines.append("  └" + "─" * 96)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Per-method text + CSV report
# ─────────────────────────────────────────────────────────────────────────────

def write_method_report(method_name: str,
                        trades_df: pd.DataFrame,
                        equity_df: pd.DataFrame,
                        audited_df: pd.DataFrame,
                        output_dir: Path) -> str:
    lines: List[str] = []
    lines.append(_sec(f"STRATEGY REPORT — {method_name.upper()}"))
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    if trades_df.empty or audited_df.empty:
        lines.append("  No trades were generated.\n")
        txt = '\n'.join(lines)
        (output_dir / f'{method_name}_report.txt').write_text(txt)
        return txt

    n = len(audited_df)
    n_good    = (audited_df['signal_sense'] == 'GOOD').sum()
    n_marg    = (audited_df['signal_sense'] == 'MARGINAL').sum()
    n_sus     = (audited_df['signal_sense'] == 'SUSPECT').sum()
    win_rate  = (audited_df['net_return_pct'] > 0).mean()
    avg_ret   = audited_df['net_return_pct'].mean()
    total_pnl = audited_df['pnl_usd'].sum()
    avg_hold  = audited_df['hold_minutes'].mean()
    med_hold  = audited_df['hold_minutes'].median()

    # ── Overall ──
    lines.append(_sec("OVERALL PERFORMANCE", '-'))
    lines.append(f"  Total trades        : {n}")
    lines.append(f"  Win rate            : {win_rate*100:.1f}%")
    lines.append(f"  Avg net return/trade: {avg_ret:.4f}%")
    lines.append(f"  Total P&L           : ${total_pnl:,.2f}")
    lines.append(f"  Avg hold time       : {avg_hold:.1f} min  (median {med_hold:.1f} min)")
    lines.append(f"  Avg relevance score : {audited_df['relevance_score'].mean():.3f}")
    lines.append(f"\n  Signal quality breakdown:")
    lines.append(f"    GOOD    (rel≥0.4 & |influence|>0.0002) : {n_good:4d}  ({n_good/n*100:.0f}%)")
    lines.append(f"    MARGINAL (0.15 ≤ rel < 0.4)            : {n_marg:4d}  ({n_marg/n*100:.0f}%)")
    lines.append(f"    SUSPECT  (rel < 0.15)                  : {n_sus:4d}  ({n_sus/n*100:.0f}%)")

    # P&L by quality
    for label in ['GOOD', 'MARGINAL', 'SUSPECT']:
        sub = audited_df[audited_df['signal_sense'] == label]
        if not sub.empty:
            wr  = (sub['net_return_pct'] > 0).mean()
            pnl = sub['pnl_usd'].sum()
            lines.append(
                f"    → {label:8s}: win={wr*100:.0f}%  P&L=${pnl:,.0f}"
            )

    # ── Hold time distribution ──
    lines.append(_sec("HOLD TIME DISTRIBUTION", '-'))
    bins   = [0, 5, 15, 30, 60, 120, 360, float('inf')]
    labels = ['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2-6h', '>6h']
    hist = pd.cut(audited_df['hold_minutes'], bins=bins, labels=labels).value_counts().sort_index()
    for bucket, cnt in hist.items():
        pct = cnt / n * 100
        bar = '█' * int(pct / 2)
        lines.append(f"  {bucket:8s}: {cnt:4d} ({pct:5.1f}%)  {bar}")

    # ── Per-ticker breakdown ──
    lines.append(_sec("PER-TICKER BREAKDOWN", '-'))
    hdr = (f"  {'Ticker':6s} {'#':>4s}  {'Win%':>5s}  {'AvgRet%':>8s}  "
           f"{'AvgHold':>8s}  {'P&L':>10s}  {'AvgRel':>7s}")
    lines.append(hdr)
    lines.append("  " + "-" * 65)
    for ticker, grp in audited_df.groupby('ticker'):
        wr  = (grp['net_return_pct'] > 0).mean() * 100
        ar  = grp['net_return_pct'].mean()
        ahl = grp['hold_minutes'].mean()
        pnl = grp['pnl_usd'].sum()
        arl = grp['relevance_score'].mean()
        lines.append(
            f"  {ticker:6s} {len(grp):4d}  {wr:5.0f}%  {ar:+8.4f}%  "
            f"{ahl:8.1f}m  ${pnl:>9,.0f}  {arl:7.3f}"
        )

    # ── Full chronological trade log ──
    lines.append(_sec("COMPLETE TRADE LOG  (chronological, all trades)", '-'))
    lines.append(
        "  Format: [entry → exit]  hold=Xmin  TICKER DIR  "
        "pos=$X  influence=X  gross/net%  P&L  capital | tweet\n"
    )
    for _, row in audited_df.iterrows():
        lines.extend(_fmt_trade_block(row))

    # ── Top winners / losers ──
    lines.append(_sec("TOP 10 WINNERS", '-'))
    for _, row in audited_df.nlargest(10, 'net_return_pct').iterrows():
        lines.extend(_fmt_trade_block(row))

    lines.append(_sec("TOP 10 LOSERS", '-'))
    for _, row in audited_df.nsmallest(10, 'net_return_pct').iterrows():
        lines.extend(_fmt_trade_block(row))

    # ── GOOD signals ──
    lines.append(_sec("GOOD SIGNALS (relevance≥0.4, strong influence)", '-'))
    good = audited_df[audited_df['signal_sense'] == 'GOOD']
    if good.empty:
        lines.append("  None above threshold.")
    else:
        for _, row in good.iterrows():
            lines.extend(_fmt_trade_block(row))

    # ── SUSPECT signals ──
    lines.append(_sec("SUSPECT SIGNALS (tweet likely unrelated to ticker)", '-'))
    sus = audited_df[audited_df['signal_sense'] == 'SUSPECT']
    if sus.empty:
        lines.append("  None — all trades pass the relevance filter.  ✓")
    else:
        for _, row in sus.iterrows():
            lines.extend(_fmt_trade_block(row))

    # ── Relevance histogram ──
    lines.append(_sec("RELEVANCE SCORE DISTRIBUTION", '-'))
    rbins   = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
    rlabels = ['0.0–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.5', '0.5–0.7', '0.7+']
    rhist = pd.cut(audited_df['relevance_score'], bins=rbins, labels=rlabels).value_counts().sort_index()
    for bk, cnt in rhist.items():
        pct = cnt / n * 100
        bar = '█' * int(pct / 2)
        lines.append(f"  {bk:8s}: {cnt:4d} ({pct:5.1f}%)  {bar}")

    txt = '\n'.join(lines) + '\n'
    (output_dir / f'{method_name}_report.txt').write_text(txt)
    print(f"  Text report  → {output_dir / f'{method_name}_report.txt'}")

    # ── Full CSV trade log ──
    csv_path = output_dir / f'{method_name}_trade_log.csv'
    audited_df.to_csv(csv_path, index=False)
    print(f"  Trade log CSV→ {csv_path}")

    return txt


# ─────────────────────────────────────────────────────────────────────────────
# Strategy runners
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(method_name: str,
                 scorer,
                 tweets_df: pd.DataFrame,
                 price_data: Dict,
                 returns_df: pd.DataFrame,
                 tickers: List[str],
                 output_dir: Path,
                 rf: TickerRelevanceFilter,
                 backtest_tweets: Optional[pd.DataFrame] = None) -> Dict:
    """Generic runner: backtest → audit → text report → quantstats."""
    bt_tweets = backtest_tweets if backtest_tweets is not None else tweets_df

    backtester = TweetStrategyBacktester(
        initial_capital=INITIAL_CAPITAL,
        holding_period_minutes=HOLDING_PERIOD_MINUTES,
        apply_transaction_costs=True,
        max_trades_per_day=10,
        use_time_filter=False,
        use_regime_detection=False,
        use_advanced_sizing=False,
    )

    trades_df = equity_df = pd.DataFrame()
    final_capital = INITIAL_CAPITAL
    for threshold in [0.0002, 0.0001, 5e-5, 0.0]:
        trades_df, equity_df, final_capital = backtester.backtest_strategy(
            tweets_df=bt_tweets,
            scorer=scorer,
            tickers=tickers,
            price_data=price_data,
            return_horizon='30m',
            min_score_threshold=threshold,
        )
        if len(trades_df) > 0:
            print(f"  {len(trades_df)} trades at threshold={threshold}")
            break

    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    print(f"  Total return : {total_return*100:.2f}%")

    audited = audit_trades(trades_df, bt_tweets, rf)
    write_method_report(method_name, trades_df, equity_df, audited, output_dir)
    generate_quantstats(method_name, equity_df, price_data, output_dir)

    return {
        'method':        method_name,
        'trades_df':     trades_df,
        'equity_df':     equity_df,
        'audited_df':    audited,
        'final_capital': final_capital,
        'total_return':  total_return,
    }


def run_sentiment(tweets_df, price_data, returns_df, tickers, output_dir, rf) -> Dict:
    print(_sec("RUNNING SENTIMENT STRATEGY"))
    try:
        scorer = SentimentScorer(FinancialSentimentAnalyzer())
    except Exception as e:
        print(f"  Scorer init error: {e}")
        return {}
    return run_strategy('sentiment', scorer, tweets_df, price_data,
                        returns_df, tickers, output_dir, rf)


def run_correlation(tweets_df, price_data, returns_df, tickers, output_dir, rf) -> Dict:
    print(_sec("RUNNING CORRELATION STRATEGY"))
    scorer = CorrelationDiscovery(
        train_ratio=0.7, p_value_max=0.05,
        min_abs_correlation=0.02, use_price_features=False,
    )
    scorer.discover_relationships(tweets_df, returns_df, tickers, return_horizon='30m')

    print("\n  Feature importance (top 5 per ticker):")
    for t in tickers:
        scorer.print_feature_importance(t, top_n=5)

    test_tweets = scorer.get_backtest_tweets(tweets_df)
    print(f"\n  Test-period tweets: {len(test_tweets)}")
    if len(test_tweets) == 0:
        print("  No test-period tweets — skipping.")
        return {}

    # Save feature importance
    fi = scorer.get_all_feature_importance()
    if not fi.empty:
        fi.to_csv(output_dir / 'correlation_feature_importance.csv', index=False)
        print(f"  Feature importance → {output_dir / 'correlation_feature_importance.csv'}")

    return run_strategy('correlation', scorer, tweets_df, price_data,
                        returns_df, tickers, output_dir, rf,
                        backtest_tweets=test_tweets)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison(results: Dict, tweets_df: pd.DataFrame,
                     price_data: Dict, output_dir: Path):
    lines: List[str] = []
    lines.append(_sec("STRATEGY COMPARISON REPORT"))
    lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    # ── Tweet coverage ──
    lines.append(_sec("TWEET COVERAGE (after relevance filter)", '-'))
    lines.append(f"  Unique tweets : {tweets_df['tweet_id'].nunique():,}")
    lines.append(f"  Rows          : {len(tweets_df):,}\n")
    grp = tweets_df.groupby('ticker')
    cov = grp['tweet_id'].count().rename('tweets')
    if 'relevance_score' in tweets_df.columns:
        cov = pd.concat([cov, grp['relevance_score'].mean().rename('avg_rel').round(3)], axis=1)
    lines.append(cov.sort_values('tweets', ascending=False).to_string())

    # ── Performance table ──
    lines.append(_sec("PERFORMANCE SUMMARY", '-'))
    perf = []
    for name, r in results.items():
        if not r:
            continue
        a = r.get('audited_df', pd.DataFrame())
        if a.empty:
            continue
        perf.append({
            'Method':            name,
            'Trades':            len(a),
            'Win%':              round((a['net_return_pct'] > 0).mean() * 100, 1),
            'AvgRet%':           round(a['net_return_pct'].mean(), 4),
            'AvgHold(min)':      round(a['hold_minutes'].mean(), 1),
            'TotalPnL($)':       round(a['pnl_usd'].sum(), 2),
            'TotalReturn%':      round(r['total_return'] * 100, 2),
            'AvgRelevance':      round(a['relevance_score'].mean(), 3),
            'GOOD%':             round((a['signal_sense']=='GOOD').mean()*100, 0),
            'SUSPECT%':          round((a['signal_sense']=='SUSPECT').mean()*100, 1),
        })
    pf = pd.DataFrame(perf)
    if not pf.empty:
        lines.append(pf.to_string(index=False))
        pf.to_csv(output_dir / 'comparison_table.csv', index=False)

    # ── Signal quality per method ──
    lines.append(_sec("SIGNAL QUALITY BREAKDOWN", '-'))
    for name, r in results.items():
        if not r:
            continue
        a = r.get('audited_df', pd.DataFrame())
        if a.empty:
            continue
        total = len(a)
        lines.append(f"\n  {name.upper()}  ({total} trades)")
        for label in ['GOOD', 'MARGINAL', 'SUSPECT']:
            sub = a[a['signal_sense'] == label]
            if sub.empty:
                continue
            wr  = (sub['net_return_pct'] > 0).mean() * 100
            pnl = sub['pnl_usd'].sum()
            lines.append(
                f"    {label:8s}: {len(sub):4d} ({len(sub)/total*100:5.1f}%)  "
                f"win={wr:.0f}%  P&L=${pnl:,.0f}"
            )

    # ── Hold-time comparison ──
    lines.append(_sec("HOLD TIME COMPARISON", '-'))
    for name, r in results.items():
        if not r:
            continue
        a = r.get('audited_df', pd.DataFrame())
        if a.empty:
            continue
        lines.append(
            f"  {name:15s}  mean={a['hold_minutes'].mean():.1f}m  "
            f"median={a['hold_minutes'].median():.1f}m  "
            f"min={a['hold_minutes'].min():.1f}m  "
            f"max={a['hold_minutes'].max():.1f}m"
        )

    # ── Equity curves comparison chart ──
    if MPL_AVAILABLE and results:
        try:
            fig, ax = plt.subplots(figsize=(13, 6))
            for name, r in results.items():
                eq = r.get('equity_df', pd.DataFrame())
                if eq.empty:
                    continue
                eq = eq.copy()
                eq['time'] = pd.to_datetime(eq['time'])
                eq = eq.set_index('time').sort_index()
                norm = (eq['equity'] / INITIAL_CAPITAL - 1) * 100
                ax.plot(norm.index, norm.values, linewidth=1.8, label=name)
            if 'SPY' in price_data:
                spy_d = price_data['SPY']['close'].resample('D').last().ffill()
                spy_n = (spy_d / spy_d.iloc[0] - 1) * 100
                ax.plot(spy_n.index, spy_n.values, linewidth=1,
                        linestyle='--', color='grey', alpha=0.7, label='SPY (ref)')
            ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
            ax.set_title('Equity Curves — Strategy Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Return from start (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            cmp_chart = output_dir / 'comparison_equity.png'
            plt.savefig(cmp_chart, dpi=150)
            plt.close(fig)
            lines.append(_sec("CHARTS", '-'))
            lines.append(f"  Equity comparison chart → {cmp_chart}")
        except Exception as e:
            print(f"  Comparison chart failed: {e}")

    # ── Interpretation ──
    lines.append(_sec("INTERPRETATION GUIDE", '-'))
    lines.append(textwrap.dedent("""
  Signal Quality Labels
  ─────────────────────
  GOOD     relevance≥0.4 AND |influence|>0.0002
           Tweet clearly discusses a topic that moves this ticker.
           These are the highest-confidence trades.

  MARGINAL relevance 0.15–0.4
           Tweet is plausibly related but not highly specific.
           Most trades will fall here after relevance filtering.

  SUSPECT  relevance<0.15
           Tweet does not obviously relate to this ticker.
           These are noise trades.  If SUSPECT% > 5%, tighten
           min_relevance_score in TickerRelevanceFilter.

  Relevance Score
  ───────────────
  Computed by keyword matching against per-ticker term dictionaries.
  Required terms gate entry (score=0 if none match).
  Primary terms weight 2×, secondary terms weight 1×.
  Score capped at 1.0; direct ticker symbol mention adds +0.3.

  Hold Time
  ─────────
  Set by HOLDING_PERIOD_MINUTES in config/settings.py (currently
  {hp} minutes).  Actual hold may differ if a later tweet triggers
  an early position close.
    """.format(hp=HOLDING_PERIOD_MINUTES)))

    txt = '\n'.join(lines) + '\n'
    (output_dir / 'comparison_report.txt').write_text(txt)
    print(f"\nComparison report → {output_dir / 'comparison_report.txt'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['sentiment', 'correlation', 'both'],
                        default='both')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--min-relevance', type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else OUTPUT_DIR / 'strategy_report'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    tweets_df, price_data, returns_df, tickers = load_data(args.min_relevance)

    if tweets_df.empty:
        print("No relevant tweets — aborting.")
        return

    # Merge forward returns into tweets so scorers can access them
    if not returns_df.empty and 'tweet_id' in returns_df.columns:
        tweets_df = pd.merge(tweets_df, returns_df, on='tweet_id', how='left')

    rf = TickerRelevanceFilter(min_relevance_score=args.min_relevance)
    results: Dict = {}

    if args.method in ('sentiment', 'both'):
        results['sentiment'] = run_sentiment(tweets_df, price_data, returns_df,
                                             tickers, out_dir, rf)

    if args.method in ('correlation', 'both'):
        results['correlation'] = run_correlation(tweets_df, price_data, returns_df,
                                                 tickers, out_dir, rf)

    write_comparison(results, tweets_df, price_data, out_dir)

    print(_sec("ALL DONE"))
    print(f"Files in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:<45s}  {size/1024:>7.1f} KB")


if __name__ == '__main__':
    main()
