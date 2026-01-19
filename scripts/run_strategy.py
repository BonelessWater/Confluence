"""
TRUMP TWEET TRADING STRATEGY - Main Entry Point

This script provides access to all strategy components:
1. Robust strategy with statistical controls (recommended)
2. Tweet signal report for manual checking
3. Feature precomputation

Usage:
    python scripts/run_strategy.py                    # Full analysis
    python scripts/run_strategy.py --report-only     # Just the tweet report
    python scripts/run_strategy.py --ticker QQQ      # Analyze different ticker
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def run_robust_analysis(ticker='SPY'):
    """Run the robust strategy with statistical controls."""
    print("="*70)
    print("ROBUST STRATEGY ANALYSIS")
    print("="*70)

    from final_robust_strategy import FinalRobustStrategy

    strategy = FinalRobustStrategy(
        initial_capital=100000,
        max_trades_per_day=2
    )
    strategy.run_comparison(ticker)


def run_report(ticker='SPY'):
    """Generate the comprehensive tweet-signal report."""
    print("="*70)
    print("GENERATING TWEET-SIGNAL REPORT")
    print("="*70)

    from comprehensive_tweet_report import (
        load_all_data, calculate_all_returns, detect_keywords,
        generate_signals, calculate_signal_accuracy, generate_reports
    )

    df, market_pl = load_all_data(ticker)
    df = calculate_all_returns(df, market_pl)
    df = detect_keywords(df)
    df = generate_signals(df)
    df = calculate_signal_accuracy(df)
    generate_reports(df, ticker)

    print(f"\n[OK] Reports generated:")
    print(f"     - output/{ticker}_comprehensive_tweet_report.csv")
    print(f"     - output/{ticker}_comprehensive_tweet_report.html")


def main():
    parser = argparse.ArgumentParser(description='Trump Tweet Trading Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--report-only', action='store_true', help='Only generate report')
    parser.add_argument('--analysis-only', action='store_true', help='Only run robust analysis')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("TRUMP TWEET TRADING STRATEGY")
    print("="*70)
    print(f"Ticker: {args.ticker}")
    print("="*70)

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.report_only:
        run_report(args.ticker)
    elif args.analysis_only:
        run_robust_analysis(args.ticker)
    else:
        run_robust_analysis(args.ticker)
        run_report(args.ticker)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print(f"  output/{args.ticker}_full_signal_analysis.csv      (statistical analysis)")
    print(f"  output/{args.ticker}_comprehensive_tweet_report.html (interactive report)")


if __name__ == "__main__":
    main()
