"""
Compare two exit strategies across two scorers (correlation, sentiment):
  1. event_driven  — current implementation (exit when next tweet arrives after 5-min window)
  2. fixed_time    — exit exactly at entry_time + 5 minutes using real prices

Output structure:
  output/exit_mode_comparison/
    correlation/
      event_driven/  quantstats_report.html, trades.csv, equity_curve.csv
      fixed_time/    quantstats_report.html, trades.csv, equity_curve.csv
      equity_curve_overlay.png
    sentiment/
      event_driven/  ...
      fixed_time/    ...
      equity_curve_overlay.png
    comparison_summary.csv   (all 4 rows)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config.settings import INITIAL_CAPITAL, HOLDING_PERIOD_MINUTES, OUTPUT_DIR
from src.tweet_ticker_analysis.backtest_tweet_strategies import TweetStrategyBacktester
from src.tweet_ticker_analysis.correlation_discovery import CorrelationDiscovery
from src.tweet_ticker_analysis.sentiment_analyzer import FinancialSentimentAnalyzer, SentimentScorer
from scripts.run_all_tweet_ticker_methods import load_data

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    print("Warning: quantstats not installed — HTML tearsheets will be skipped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_returns_series(equity_df: pd.DataFrame) -> pd.Series:
    """Convert equity curve to daily return series suitable for quantstats."""
    eq = equity_df.copy()
    eq['time'] = pd.to_datetime(eq['time'])
    eq = eq.set_index('time').sort_index()
    daily = eq['equity'].resample('D').last().ffill().dropna()
    return daily.pct_change().dropna()


def run_backtest(scorer, scorer_name, tweets_df, price_data, valid_tickers,
                 returns_df, exit_mode: str, method_dir: Path) -> dict:
    """Run one backtest pass and save results."""
    print(f"\n{'='*70}")
    print(f"SCORER: {scorer_name.upper()}  |  EXIT MODE: {exit_mode.upper()}")
    print(f"{'='*70}")

    # Per-scorer thresholds (same as run_all_tweet_ticker_methods)
    thresholds = {
        'correlation': 0.0002,
        'sentiment':   0.0005,
    }
    min_threshold = thresholds.get(scorer_name, 0.0002)

    backtester = TweetStrategyBacktester(
        initial_capital=INITIAL_CAPITAL,
        holding_period_minutes=HOLDING_PERIOD_MINUTES,
        apply_transaction_costs=True,
        max_trades_per_day=3,
        use_time_filter=True,
        min_hour=12,
        max_hour=14,
        use_regime_detection=True,
        use_advanced_sizing=True,
        exit_mode=exit_mode,
    )

    trades_df = pd.DataFrame()
    equity_df = pd.DataFrame()
    final_capital = INITIAL_CAPITAL

    for threshold in [min_threshold, min_threshold * 0.5, 1e-6]:
        trades_df, equity_df, final_capital = backtester.backtest_strategy(
            tweets_df=tweets_df,
            scorer=scorer,
            tickers=valid_tickers,
            price_data=price_data,
            return_horizon='30m',
            min_score_threshold=threshold,
            returns_df=returns_df,
        )
        if len(trades_df) > 0:
            break

    method_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(method_dir / 'trades.csv', index=False)
    equity_df.to_csv(method_dir / 'equity_curve.csv', index=False)

    num_trades = len(trades_df)
    if num_trades == 0:
        print("  No trades generated.")
        return {
            'scorer': scorer_name, 'exit_mode': exit_mode,
            'total_return': 0.0, 'final_capital': INITIAL_CAPITAL,
            'num_trades': 0, 'win_rate': 0.0, 'avg_return': 0.0,
            'avg_duration_min': 0.0, 'max_duration_min': 0.0,
            'sharpe': float('nan'), 'max_drawdown': float('nan'),
        }

    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    win_rate     = (trades_df['net_return'] > 0).mean()
    avg_return   = trades_df['net_return'].mean()
    avg_duration = trades_df['duration_minutes'].mean()
    max_duration = trades_df['duration_minutes'].max()

    print(f"\n  Total Return   : {total_return*100:.2f}%")
    print(f"  Final Capital  : ${final_capital:,.2f}")
    print(f"  Num Trades     : {num_trades}")
    print(f"  Win Rate       : {win_rate*100:.1f}%")
    print(f"  Avg Return     : {avg_return*100:.4f}%")
    print(f"  Avg Duration   : {avg_duration:.1f} min")
    print(f"  Max Duration   : {max_duration:.1f} min")

    sharpe = float('nan')
    max_dd = float('nan')

    if QUANTSTATS_AVAILABLE and len(equity_df) > 2:
        try:
            ret_series = build_returns_series(equity_df)
            if len(ret_series) > 2:
                sharpe = qs.stats.sharpe(ret_series)
                max_dd = qs.stats.max_drawdown(ret_series)
                print(f"  Sharpe         : {sharpe:.3f}")
                print(f"  Max Drawdown   : {max_dd*100:.2f}%")

                report_path = method_dir / 'quantstats_report.html'
                qs.reports.html(
                    ret_series,
                    output=report_path.as_posix(),
                    title=f"{scorer_name.capitalize()} — {exit_mode} exit",
                )
                print(f"  Quantstats report -> {report_path}")

                # Per-run equity curve PNG
                try:
                    import matplotlib.pyplot as plt
                    eq2 = equity_df.copy()
                    eq2['time'] = pd.to_datetime(eq2['time'])
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(eq2['time'], eq2['equity'], linewidth=2)
                    ax.set_title(f"{scorer_name.capitalize()} — {exit_mode} exit")
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Equity ($)')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(method_dir / 'equity_curve.png', dpi=150)
                    plt.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"  Warning: quantstats failed: {e}")

    return {
        'scorer': scorer_name, 'exit_mode': exit_mode,
        'total_return': total_return, 'final_capital': final_capital,
        'num_trades': num_trades, 'win_rate': win_rate,
        'avg_return': avg_return, 'avg_duration_min': avg_duration,
        'max_duration_min': max_duration, 'sharpe': sharpe,
        'max_drawdown': max_dd,
    }


def overlay_plot(scorer_name: str, scorer_dir: Path):
    """Overlay event_driven vs fixed_time equity curves for one scorer."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = {'event_driven': 'steelblue', 'fixed_time': 'darkorange'}
        labels = {'event_driven': 'Event-Driven (current)', 'fixed_time': 'Fixed 5-Min Exit'}

        for mode in ['event_driven', 'fixed_time']:
            eq_path = scorer_dir / mode / 'equity_curve.csv'
            if eq_path.exists():
                eq = pd.read_csv(eq_path)
                eq['time'] = pd.to_datetime(eq['time'])
                eq = eq.sort_values('time')
                normalized = (eq['equity'] / INITIAL_CAPITAL - 1) * 100
                ax.plot(eq['time'], normalized, label=labels[mode],
                        color=colors[mode], linewidth=2)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f"{scorer_name.capitalize()} — Exit Mode Comparison (Cumulative Return %)",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return (%)')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = scorer_dir / 'equity_curve_overlay.png'
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Overlay chart -> {path}")
    except Exception as e:
        print(f"  Warning: overlay plot failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = OUTPUT_DIR / 'exit_mode_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXIT MODE COMPARISON — correlation & sentiment × event_driven & fixed_time")
    print("=" * 70)

    tweets_df, price_data, returns_df, validator, valid_tickers = load_data()

    if 'tweet_id' in returns_df.columns and 'tweet_id' in tweets_df.columns:
        tweets_df = pd.merge(tweets_df, returns_df, on='tweet_id', how='left')

    # ------------------------------------------------------------------
    # Build scorers
    # ------------------------------------------------------------------
    scorers = {}

    # Correlation — needs training + test split
    print("\nTraining correlation scorer...")
    corr_scorer = CorrelationDiscovery(
        train_ratio=0.7, p_value_max=0.05,
        min_abs_correlation=0.02, use_price_features=False,
    )
    corr_scorer.discover_relationships(tweets_df, returns_df, valid_tickers, return_horizon='30m')
    corr_backtest_tweets = corr_scorer.get_backtest_tweets(tweets_df)
    print(f"  Correlation backtest tweets: {len(corr_backtest_tweets)}")
    if len(corr_backtest_tweets) > 0:
        scorers['correlation'] = (corr_scorer, corr_backtest_tweets)

    # Sentiment — zero-shot, uses all tweets
    print("\nInitialising sentiment scorer...")
    try:
        sentiment_scorer = SentimentScorer(FinancialSentimentAnalyzer())
        scorers['sentiment'] = (sentiment_scorer, tweets_df)
        print("  Sentiment scorer ready.")
    except Exception as e:
        print(f"  Skipping sentiment scorer: {e}")

    # ------------------------------------------------------------------
    # Run all combinations
    # ------------------------------------------------------------------
    all_results = []

    for scorer_name, (scorer, bt_tweets) in scorers.items():
        scorer_dir = out_dir / scorer_name
        scorer_dir.mkdir(parents=True, exist_ok=True)

        for mode in ['event_driven', 'fixed_time']:
            result = run_backtest(
                scorer=scorer,
                scorer_name=scorer_name,
                tweets_df=bt_tweets.copy(),
                price_data=price_data,
                valid_tickers=valid_tickers,
                returns_df=returns_df,
                exit_mode=mode,
                method_dir=scorer_dir / mode,
            )
            all_results.append(result)

        overlay_plot(scorer_name, scorer_dir)

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FULL COMPARISON SUMMARY")
    print(f"{'='*70}")

    rows = []
    for r in all_results:
        rows.append({
            'Scorer': r['scorer'],
            'Exit Mode': r['exit_mode'],
            'Total Return (%)': f"{r['total_return']*100:.2f}",
            'Num Trades': r['num_trades'],
            'Win Rate (%)': f"{r['win_rate']*100:.1f}",
            'Avg Return/Trade (%)': f"{r['avg_return']*100:.4f}",
            'Avg Duration (min)': f"{r['avg_duration_min']:.1f}",
            'Max Duration (min)': f"{r['max_duration_min']:.1f}",
            'Sharpe': f"{r['sharpe']:.3f}" if not np.isnan(r['sharpe']) else 'N/A',
            'Max Drawdown (%)': f"{r['max_drawdown']*100:.2f}" if not np.isnan(r['max_drawdown']) else 'N/A',
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(out_dir / 'comparison_summary.csv', index=False)

    print(f"\nAll outputs in: {out_dir}")
    for scorer_name in scorers:
        for mode in ['event_driven', 'fixed_time']:
            print(f"  {scorer_name}/{mode}/quantstats_report.html")
    print("  correlation/equity_curve_overlay.png")
    print("  sentiment/equity_curve_overlay.png")
    print("  comparison_summary.csv")


if __name__ == '__main__':
    main()
