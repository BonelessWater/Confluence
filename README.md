# L1 Market Data Consolidation & Predictive Power Analysis

This repository provides tools for analyzing the predictive power of different Level 1 (L1) market data consolidation methods across multiple stocks.

## Overview

When analyzing stock prices across multiple exchanges, different consolidation methods can be used to estimate the true price given bid-ask spreads from each exchange. This project evaluates and compares four different methods:

1. **Consensus Mid**: Simple midpoint average across all exchanges
2. **Consensus Microprice**: Volume-weighted microprice
3. **EMA Estimate**: Exponential Moving Average filter for smoothing
4. **Kalman Filter**: Kalman filtering for optimal state estimation

## Directory Structure

```
confluence/
├── src/                              # Source code
│   ├── analyze.py                    # Main analysis script
│   └── L1_transform/                 # Market data consolidation methods
│       ├── __init__.py
│       ├── per_exchange_state.py     # Multi-exchange order book tracking
│       ├── consensus_mid.py          # Simple midpoint method
│       ├── consensus_microprice.py   # Volume-weighted microprice
│       ├── ema_estimate.py           # EMA-based estimation
│       ├── kalman_filter.py          # Kalman filter estimation
│       ├── metrics.py                # Metrics calculation & ranking
│       └── analyze_l1_methods.py     # Legacy method analysis
│
├── data/                             # Input data
│   ├── market_data.parquet           # Multi-stock L1 market data
│   └── A.csv                         # Single stock data (legacy)
│
├── output/                           # Results and visualizations
│   ├── l1_multi_stock_summary.csv    # Performance metrics summary
│   └── l1_multi_stock_comparison.png # Comparative visualizations
│
├── tweet_analysis/                   # Twitter sentiment analysis (separate project)
│   └── [files...]
│
└── README.md                         # This file
```

## Installation

### Requirements
- Python 3.8+
- pandas
- polars
- numpy
- matplotlib
- seaborn

### Setup
```bash
pip install pandas polars numpy matplotlib seaborn
```

## Usage

### Basic Analysis

Run the main analysis on your parquet file:

```bash
cd src
python analyze.py --parquet ../data/market_data.parquet
```

### Advanced Options

```bash
python analyze.py --parquet <path_to_parquet> \
                  --output <output_dir> \
                  --max-rows <n> \
                  --tickers TICKER1,TICKER2
```

**Arguments:**
- `--parquet` (required): Path to parquet file with market data
- `--output`: Output directory for results (default: `../output`)
- `--max-rows`: Max rows to process per stock (default: 50000)
- `--tickers`: Comma-separated tickers to analyze (default: all)

### Example

```bash
python analyze.py --parquet ../data/market_data.parquet \
                  --tickers AAPL,MSFT,GOOGL \
                  --output ../output
```

## Output

The analysis produces:

1. **Console Output**: Summary statistics and rankings by stock
2. **CSV File**: `l1_multi_stock_summary.csv` with detailed metrics
3. **Visualization**: `l1_multi_stock_comparison.png` with 4 comparison charts

### Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices (lower is better)
- **RMSE**: Root mean square error (penalizes larger errors more)
- **MAPE**: Mean absolute percentage error (% difference)
- **R-squared**: Proportion of variance explained (0-1, higher is better)
- **Correlation**: Pearson correlation between prediction and actual
- **Time (us)**: Computation time in microseconds (lower is better)

### Ranking Formula

Overall score = 50% × Prediction Error + 35% × Quality + 15% × Efficiency

Methods are ranked 0-10 on each metric, then combined with weights:
- Prediction Error (MAE): Lower is better
- Prediction Quality (R²): Higher is better
- Computational Efficiency: Faster is better

## Method Details

### 1. Consensus Mid
The simple average of best bid/ask midpoints across all exchanges.

**Pros**: Simple, fast
**Cons**: Doesn't account for volume, may be less accurate

### 2. Consensus Microprice
Volume-weighted average of bid-ask midpoints.

**Formula**: `(ask_price × bid_volume + bid_price × ask_volume) / (bid_volume + ask_volume)`

**Pros**: Accounts for order book depth
**Cons**: Still non-adaptive

### 3. EMA Estimate
Exponential Moving Average applied to consensus mid prices.

**Pros**: Smooth, reactive to recent changes, fast
**Cons**: Lags behind actual price movements

### 4. Kalman Filter
Optimal linear estimator for dynamic systems.

**Pros**: Mathematically optimal for linear systems, adaptive
**Cons**: More computationally intensive, requires tuning

## Module Reference

### analyze.py
Main entry point. Orchestrates the analysis pipeline:
1. Loads parquet file
2. Processes each stock
3. Generates predictions
4. Calculates metrics
5. Creates visualizations

### L1_transform/

#### per_exchange_state.py
`MultiExchangeBook`: Tracks bid/ask quotes from multiple exchanges

#### consensus_mid.py
`compute_consensus_mid(book, timestamp)`: Simple midpoint consolidation

#### consensus_microprice.py
`compute_consensus_micro(book, timestamp)`: Volume-weighted consolidation

#### ema_estimate.py
`EMAEfficientPrice(alpha)`: Exponential moving average filter

#### kalman_filter.py
`KalmanEfficientPrice(process_var, obs_var)`: Kalman filter

#### metrics.py
`PredictionMetrics`: Utility class for calculating accuracy, quality, and ranking metrics

Functions:
- `calculate_accuracy_metrics()`: MAE, RMSE, MAPE, max error
- `calculate_quality_metrics()`: R², correlation
- `measure_computation_time()`: Benchmark a function
- `rank_methods()`: Rank methods by weighted score
- `evaluate_method()`: Comprehensive evaluation

## Data Format

Expected parquet schema:
```
- Date: int32
- Timestamp: string (ISO format or similar)
- EventType: string (contains 'BID' or 'ASK')
- Ticker: string
- Price: float32
- Quantity: int32
- Exchange: string
- Conditions: string (optional)
```

## Examples

### Compare specific stocks
```bash
python analyze.py --parquet ../data/market_data.parquet --tickers AAPL,MSFT
```

### Process with custom row limit
```bash
python analyze.py --parquet ../data/market_data.parquet --max-rows 100000
```

### Save to custom output directory
```bash
python analyze.py --parquet ../data/market_data.parquet --output /path/to/results
```

## Implementation Notes

### Design Principles
- **Modularity**: Each method is in its own file with a clear interface
- **Reusability**: Methods can be imported and used independently
- **Scalability**: Supports multiple stocks in a single parquet file
- **Extensibility**: Easy to add new methods by following the same pattern

### Performance Considerations
- The analysis processes up to `--max-rows` per stock to manage memory
- Snapshots are created by grouping nearby timestamps
- Computation times include 500 iterations for statistical accuracy

### Code Organization
- Methods are pure functions with clear inputs/outputs
- State machines (EMA, Kalman) have update() methods
- Metrics are calculated using vectorized numpy operations
- Results are saved to CSV and PNG for easy sharing

## Testing & Validation

To validate the setup:
```bash
cd src
python -c "from L1_transform import compute_consensus_mid; print('Import successful')"
```

## Troubleshooting

**ImportError when running analyze.py**
- Ensure you're in the `src/` directory
- Check that `L1_transform/` exists with all method files

**No tickers found**
- Verify the parquet file has a 'Ticker' column
- Check that ticker symbols match if using `--tickers` filter

**Memory error with large files**
- Reduce `--max-rows` value
- Process specific tickers with `--tickers` option

**No valid predictions generated**
- Check that data has valid bid/ask prices (> 0)
- Ensure multiple exchanges are present per symbol
- Verify timestamps are properly formatted

## License

This project is part of the confluence research repository.

## References

- Kalman, R. E. (1960). "A new approach to linear filtering and prediction problems"
- Chakrabarty, B., Pascual, R., & Moulton, P. C. (2015). "Liquidity and Instability"
