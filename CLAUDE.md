# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository contains tools for analyzing Level 1 (L1) market data consolidation across multiple exchanges and evaluating different price estimation methods. It also includes a separate tweet analysis module for analyzing temporal clustering of tweets.

## Core Architecture

### L1 Market Data Consolidation System

The system processes real-time bid/ask quotes from multiple exchanges (NYSE, NASDAQ, CBOE, etc.) and estimates the "true price" using four different consolidation methods:

1. **Consensus Mid** - Simple average of midpoints across exchanges
2. **Consensus Microprice** - Volume-weighted average using order book depth
3. **EMA Estimate** - Exponential Moving Average for noise filtering
4. **Kalman Filter** - Optimal state estimation with adaptive filtering

**Key Components:**

- `MultiExchangeBook` (per_exchange_state.py:13) - Tracks best bid/ask from each exchange
- Consolidation methods - Each in separate file, pure functions or stateful classes
- `PredictionMetrics` (metrics.py) - Calculates MAE, RMSE, R², correlation, computation time
- `analyze.py` - Main orchestrator that processes multiple stocks and generates reports

**Data Flow:**
```
Raw Parquet → Extract per-stock → Create L1 snapshots → Generate predictions
→ Compare to actual next price → Calculate metrics → Rank methods → Output CSV + PNG
```

**Snapshot Creation:**
- Groups quotes into time windows (analyze.py:100)
- Extracts best bid/ask per exchange per window (analyze.py:75)
- Requires at least 2 exchanges per snapshot to be valid

**Prediction Generation:**
- Iterates through consecutive time windows (analyze.py:134)
- Builds `MultiExchangeBook` for current window
- Applies all 4 methods to get price estimates
- Uses next window's consensus mid as "actual" price for validation
- State-based methods (EMA, Kalman) maintain state across iterations

### Project Structure

```
src/
├── analyze.py                    # Main analysis script - entry point
├── analyze_parallel.py           # Parallel processing variant (not primary)
├── debug_paths.py                # Path debugging utility
└── L1_transform/                 # Core consolidation methods package
    ├── __init__.py               # Package exports
    ├── per_exchange_state.py     # MultiExchangeBook class
    ├── consensus_mid.py          # Simple midpoint method
    ├── consensus_microprice.py   # Volume-weighted method
    ├── ema_estimate.py           # EMA filter (stateful)
    ├── kalman_filter.py          # Kalman filter (stateful)
    ├── metrics.py                # PredictionMetrics utility class
    └── analyze_l1_methods.py     # Legacy single-stock analyzer

tweet_analysis/                   # Separate module for tweet analysis
    ├── analyze_tweets.py         # Tweet clustering analysis
    ├── analyze_tweet_timestamps.py
    ├── analyze_tweet_windows.py
    └── monthly_analysis.py

data/                             # Input data (not in git)
    └── market_data.parquet       # Multi-stock L1 quotes

output/                           # Generated results (not in git)
    ├── l1_multi_stock_summary.csv
    └── l1_multi_stock_comparison.png
```

## Running the Analysis

### Basic Command

**IMPORTANT:** Always run from the `src/` directory:

```bash
cd src
python analyze.py --parquet ../data/market_data.parquet
```

The script requires being in `src/` directory for Python imports to work correctly (src is added to sys.path).

### Common Options

```bash
# Analyze specific stocks
python analyze.py --parquet ../data/market_data.parquet --tickers AAPL,MSFT,GOOGL

# Limit rows for faster processing
python analyze.py --parquet ../data/market_data.parquet --max-rows 25000

# Custom output directory
python analyze.py --parquet ../data/market_data.parquet --output /path/to/results
```

### Using the Shell Script

From repository root:
```bash
bash run_analysis.sh
```

This script:
- Changes to `src/` directory
- Runs analysis with default paths
- Verifies output files were created
- Reports file sizes and row counts

## Expected Data Format

Parquet file must have these columns:
- `Date` (int32)
- `Timestamp` (string, converted to datetime)
- `EventType` (string, contains 'BID' or 'ASK')
- `Ticker` (string)
- `Price` (float32, must be > 0)
- `Quantity` (int32, must be > 0)
- `Exchange` (string)
- `Conditions` (string, optional)

Each row represents a quote update from a specific exchange.

## Implementation Patterns

### Adding a New Consolidation Method

1. Create `src/L1_transform/my_method.py`:
```python
class MyEfficientPrice:
    def __init__(self, param=0.1):
        self.param = param
        self.state = None

    def update(self, price):
        # Estimation logic here
        self.state = price * self.param
        return self.state
```

2. Export from `src/L1_transform/__init__.py`
3. Import and use in `analyze.py` around line 130-177 in `generate_predictions()`
4. Add to metrics calculation around line 259-276

### Key Parameters

**EMA (ema_estimate.py)**
- `alpha=0.15` - Smoothing factor (15% current, 85% history)
- Effective lookback: ~7 time periods
- Tuned for 100ms-level updates

**Kalman Filter (kalman_filter.py)**
- `process_var=0.02` (Q) - Process noise variance
- `obs_var=0.5` (R) - Measurement noise variance
- Q/R ratio = 0.04 favors smoothing over reactivity

**Analysis (analyze.py)**
- `--max-rows` default: 50000 per stock
- Time windows: `Timestamp_ns * 10` creates ~100ms windows
- Computation benchmarks: 500 iterations for accuracy

### Ranking System

Overall Score = (50% × MAE) + (35% × R²) + (15% × Speed)

Methods are normalized to 0-10 scale where:
- MAE: Lower is better (higher score)
- R²: Higher is better (higher score)
- Time: Faster is better (higher score)

This weighting prioritizes accuracy > quality > speed, which is appropriate for backtesting/analysis. Adjust weights in analyze.py:434 if optimizing for real-time trading (increase speed weight).

## Output Interpretation

### Console Output
- Per-stock method rankings (best MAE first)
- Cross-stock performance averages
- Overall method ranking (0-10 scale)

### CSV File (l1_multi_stock_summary.csv)
Columns: Stock, Method, MAE, RMSE, MAPE, R-squared, Correlation, Time (us), Predictions

### Visualization (l1_multi_stock_comparison.png)
4 subplots:
1. MAE by stock and method (bar chart)
2. R-squared by stock and method (bar chart)
3. Average performance across stocks (dual-axis bar)
4. Computation time comparison (bar chart with values)

## Common Issues

**"No valid stocks could be processed"**
- Ensure parquet has 'Ticker' column
- Verify Price > 0 and Quantity > 0
- Check that multiple exchanges exist per ticker

**ImportError**
- You must run from `src/` directory: `cd src && python analyze.py ...`
- Package structure requires `src/L1_transform/__init__.py` to exist

**Memory errors**
- Reduce `--max-rows` (default 50000)
- Process specific tickers with `--tickers AAPL,MSFT`
- Use analyze_parallel.py for distributed processing (if available)

**"Insufficient snapshot data" / "Insufficient predictions generated"**
- Stock needs at least 10 valid snapshots and 10 predictions
- Check that data has multiple exchanges (2+ required)
- Verify timestamps are properly formatted and sequential

## Method Selection Guide

**For Real-Time Trading:** Use EMA Estimate or Kalman Filter (0.3-0.7 μs)
**For Maximum Accuracy:** Use Consensus Mid or Kalman Filter (best MAE/R²)
**For Simplicity/Auditability:** Use Consensus Mid (no parameters, easy to explain)
**For Liquidity Awareness:** Use Consensus Microprice (incorporates order book depth)

## Tweet Analysis Module (Separate)

Located in `tweet_analysis/` - analyzes temporal clustering of tweets.

**Key insight:** By consolidating overlapping 5-minute windows following tweets, can achieve ~90% reduction in data fetching requirements.

This module is independent of the L1 market data analysis and uses different input files (trump_tweets_20251026.txt).

## Development Notes

- All methods are modular - can be imported and used independently
- State machines (EMA, Kalman) use `.update()` interface
- Metrics use vectorized numpy operations for speed
- Path handling uses pathlib and supports both relative (from base_dir) and absolute paths
- analyze.py:293-311 has complex path resolution logic for parquet/output arguments
