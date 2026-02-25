> [!WARNING]
> **DEPRECATED** — This was the initial implementation log for the tweet-ticker scoring system (Methods 1–5). The system is now fully documented in [PIPELINE_README.md](PIPELINE_README.md) and [README.md](README.md). Trade output columns documented here are also out of date (see README for current schema).

# Tweet-Ticker Relationship Discovery Implementation Summary

## Overview

This implementation provides a comprehensive system for discovering relationships between Trump's tweets and stock tickers using 5 different NLP/machine learning methods, and backtesting each as a trading strategy with quantstats equity curves.

## What Was Implemented

### 1. Core Analysis Modules (`src/tweet_ticker_analysis/`)

#### Method 1: Correlation Discovery (`correlation_discovery.py`)
- Computes correlation matrix between tweet embeddings/features and ticker returns
- Uses Pearson correlation, mutual information, and statistical significance testing
- Fast and interpretable baseline method

#### Method 2: Bag-of-Words Scorer (`bag_of_words_scorer.py`)
- Extracts keywords/n-grams (unigrams, bigrams) from tweets
- Scores each keyword-ticker pair by mean return, t-statistic, hit rate, volatility impact
- Uses TF-IDF vectorization for feature extraction
- Interpretable keyword-based approach

#### Method 3: Embedding Scorer (`embedding_scorer.py`)
- Uses existing tweet embeddings for similarity matching and regression
- Trains Ridge regression model per ticker: `embedding → forward_return`
- Also supports k-NN similarity-based scoring
- Leverages semantic relationships captured in embeddings

#### Method 4: LLM Scorer (`llm_scorer.py`)
- Uses Large Language Models (GPT-3.5/GPT-4) for semantic analysis
- Zero-shot scoring: analyzes tweet content and rates ticker-specific influence
- Requires OpenAI API key (optional - gracefully handles missing key)
- Captures complex semantic relationships and context

#### Method 5: Ensemble Scorer (`ensemble_scorer.py`)
- Combines all methods with learned weights
- Can optimize weights based on validation performance
- Most robust approach, combines strengths of all methods

### 2. Backtesting Framework (`backtest_tweet_strategies.py`)

- Specialized backtester for tweet-ticker strategies
- Uses softmax-based position sizing (similar to existing backtest engine)
- Applies realistic transaction costs (8 bps per trade)
- Calculates returns from actual entry/exit prices (no circular logic)
- Generates trades and equity curves

### 3. Main Execution Script (`scripts/run_all_tweet_ticker_methods.py`)

- Orchestrates the entire pipeline:
  1. Loads tweet and market data
  2. Calculates forward returns at multiple horizons (5, 15, 30, 60 minutes)
  3. Trains all 5 methods
  4. Backtests each method as a trading strategy
  5. Generates quantstats equity curves and performance reports
  6. Creates comparison reports and visualizations

## File Structure

```
confluence/
├── src/
│   └── tweet_ticker_analysis/
│       ├── __init__.py
│       ├── correlation_discovery.py      # Method 1
│       ├── bag_of_words_scorer.py        # Method 2
│       ├── embedding_scorer.py           # Method 3
│       ├── llm_scorer.py                 # Method 4
│       ├── ensemble_scorer.py            # Method 5
│       ├── backtest_tweet_strategies.py  # Backtesting framework
│       ├── strategy_wrapper.py           # Strategy wrapper (for compatibility)
│       └── README.md                     # Module documentation
├── scripts/
│   └── run_all_tweet_ticker_methods.py  # Main execution script
├── output/
│   └── tweet_ticker_methods/            # Results organized by method
│       ├── correlation/
│       │   ├── trades.csv
│       │   ├── equity_curve.csv
│       │   ├── metrics.txt
│       │   ├── equity_curve_simple.png
│       │   └── quantstats_report.html
│       ├── bag_of_words/
│       ├── embedding/
│       ├── ensemble/
│       ├── method_comparison.csv
│       └── all_methods_comparison.png
└── requirements.txt                      # Updated with quantstats, scipy
```

## How to Use

### 1. Install Dependencies

```bash
pip install quantstats scipy
```

For LLM method (optional):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run All Methods

```bash
python scripts/run_all_tweet_ticker_methods.py
```

This will:
- Load data from `data/trump-truth-social-archive/data/truth_archive_with_features.parquet`
- Load market data from `data/{TICKER}.parquet` files
- Train and backtest all 5 methods
- Generate quantstats reports for each method
- Create comparison visualizations

### 3. View Results

Results are saved in `output/tweet_ticker_methods/`:

- **Per-method directories**: Each method has its own directory with:
  - `trades.csv`: All trades with entry/exit times, returns, P&L
  - `equity_curve.csv`: Equity over time
  - `metrics.txt`: Performance summary
  - `equity_curve_simple.png`: Simple matplotlib equity curve
  - `quantstats_report.html`: Detailed quantstats performance report (if quantstats installed)

- **Comparison files**:
  - `method_comparison.csv`: Side-by-side comparison of all methods
  - `all_methods_comparison.png`: Overlaid equity curves for all methods

## Output Format

### Trades CSV
```csv
entry_time,exit_time,ticker,weight,influence_score,gross_return,net_return,transaction_cost,pnl,capital
2024-01-01 10:00:00,2024-01-01 10:05:00,SPY,0.5,0.0012,0.0008,0.0000,0.0004,40.0,100040.0
```

### Equity Curve CSV
```csv
time,equity
2024-01-01 10:00:00,100000.0
2024-01-01 10:05:00,100040.0
```

### Metrics TXT
```
CORRELATION RESULTS
============================================================

method: correlation
total_return: 5.23%
final_capital: 105230.00
num_trades: 150
win_rate: 52.00%
avg_return: 0.0349%
```

## Key Features

### 1. Method Tracking
- Each method is clearly labeled and tracked
- Results are organized by method in separate directories
- Easy to compare performance across methods

### 2. Efficient Backtesting
- Uses existing return calculator (no circular logic)
- Applies realistic transaction costs
- Handles multiple tickers and time periods
- Efficient scoring with caching where possible

### 3. Organized File System
- Clear directory structure: `output/tweet_ticker_methods/{method_name}/`
- Consistent file naming across methods
- Easy to find and compare results

### 4. Quantstats Integration
- Generates professional equity curves
- Creates detailed performance reports (HTML)
- Monthly heatmaps, drawdown analysis, risk metrics
- Falls back gracefully if quantstats not installed

## Configuration

Edit `config/settings.py` to configure:

```python
TICKERS = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT', 'GLD']
HOLDING_PERIOD_MINUTES = 5
INITIAL_CAPITAL = 100000
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 3.0
```

## Method Comparison

| Method | Speed | Interpretability | Accuracy | Cost |
|--------|-------|------------------|----------|------|
| Correlation | Fast | High | Medium | Free |
| Bag-of-Words | Fast | Very High | Medium | Free |
| Embedding | Medium | Low | High | Free |
| LLM | Slow | Medium | Very High | Paid |
| Ensemble | Medium | Medium | Highest | Depends |

## Performance Metrics

Each method generates:
- **Total Return**: (Final Capital - Initial Capital) / Initial Capital
- **Number of Trades**: Total trades executed
- **Win Rate**: Percentage of profitable trades
- **Average Return per Trade**: Mean return per trade
- **Sharpe Ratio**: Risk-adjusted returns (via quantstats)
- **Maximum Drawdown**: Largest peak-to-trough decline (via quantstats)

## Notes

1. **LLM Method**: Requires OpenAI API key. If not available, the method is skipped gracefully.

2. **Data Requirements**: 
   - Tweets must have `tweet_id`, `tweet_content`, `entry_time`, `ticker` columns
   - Market data must be in parquet format with OHLCV columns
   - Embeddings should be pre-computed (from `precompute_features.py`)

3. **Return Horizons**: Methods use 30-minute forward returns by default, but can be configured.

4. **Transaction Costs**: All methods use realistic 8 bps per trade (5 bps commission + 3 bps slippage).

5. **Position Sizing**: Uses softmax-based allocation across concurrent tweets (same as existing backtest engine).

## Future Enhancements

- Add more sophisticated position sizing strategies
- Implement dynamic weight optimization for ensemble
- Add real-time scoring API
- Support for more LLM providers (Anthropic, open-source models)
- Fine-tuned models for specific ticker-tweet relationships

## Troubleshooting

**Error: "No embedding columns found"**
- Ensure `precompute_features.py` has been run to generate embeddings

**Error: "Market data file not found"**
- Check that `data/{TICKER}.parquet` files exist
- Verify ticker names match `config/settings.py`

**Warning: "LLM scoring disabled"**
- Set `OPENAI_API_KEY` environment variable to enable LLM method
- Or ignore - other methods will still run

**Quantstats reports not generated**
- Install quantstats: `pip install quantstats`
- Reports are optional - equity curves still generated with matplotlib

---

**Last Updated**: February 8, 2026
**Status**: ✅ Implementation Complete
