# Trading Strategy Backtesting & Market Data Analysis

This repository provides a comprehensive backtesting framework for trading strategies based on tweet sentiment analysis and machine learning models, along with tools for analyzing Level 1 (L1) market data consolidation methods.

## Overview

This repository contains two main systems:

1. **Trading Strategy Backtesting System**: A production-ready framework for backtesting trading strategies based on tweet sentiment analysis and ML predictions
2. **L1 Market Data Analysis**: Tools for evaluating different Level 1 market data consolidation methods

---

# Part 1: Trading Strategy Backtesting System

## System Architecture

The trading system follows a modular architecture with clear separation of concerns:

```
confluence/
├── config/                  # Centralized configuration
│   └── settings.py         # All trading parameters, costs, hyperparameters
├── src/
│   ├── models/             # Trading models (Linear, XGBoost, Attention, Ensemble)
│   ├── features/           # Feature engineering & lagging (prevents look-ahead bias)
│   ├── backtesting/        # Backtest engine, return calculator, transaction costs
│   ├── evaluation/         # Performance metrics & reporting
│   └── utils/              # Utilities
├── scripts/                # Runner scripts for backtesting
├── tests/                  # Comprehensive test suite (32 tests)
├── data/                   # Input data (tweets, embeddings, features, prices)
└── output/                 # Results organized by model and ticker
```

## Trading Strategy Logic

### Signal Generation & Position Sizing

The system uses a **softmax-based position sizing** approach:

1. **Model Predictions**: Each ML model (Linear, XGBoost, Attention, or Ensemble) generates a prediction score for each tweet/ticker
2. **Signal Transformation**: At each timestamp, predictions for concurrent tweets are transformed using softmax:
   ```python
   weights = softmax(predictions)  # Converts predictions to probability distribution
   ```
3. **Position Allocation**: Capital is allocated proportionally based on softmax weights
   - Higher predictions get larger position sizes
   - All positions at a given timestamp sum to 100% of capital

### Buy/Sell Logic

**Entry Signals:**
- **Trigger**: New tweet is received for a ticker
- **Condition**: No existing position for that ticker OR previous position has closed
- **Position Size**: Determined by softmax weight relative to other concurrent tweets
- **Entry Price**: Market price at tweet timestamp

**Exit Signals:**
- **Trigger**: Fixed time-based exit (default: 5 minutes after entry)
- **Exit Price**: Market price at exit timestamp
- **Return Calculation**: Uses actual entry/exit prices (NOT forward returns to avoid circular logic)

**Position Management:**
- Maximum one position per ticker at a time
- New tweets within holding period are skipped
- Positions automatically close after holding period
- Any remaining positions closed at end of backtest

### Transaction Costs

Realistic transaction costs are applied to all trades:
- **Commission**: 5 basis points (0.05%)
- **Slippage**: 3 basis points (0.03%)
- **Total Cost**: 8 bps per trade (16 bps round-trip)

```python
net_return = gross_return - (COMMISSION_BPS + SLIPPAGE_BPS) / 10000
```

This cost structure reduces annual returns by approximately 2-5%, providing realistic performance expectations.

## Models & Strategies

### 1. Linear Model (Baseline)
- Ridge regression with L2 regularization
- SelectKBest for feature selection (top 50 features)
- Fast inference (<1ms), interpretable
- Good baseline for comparison

### 2. XGBoost Model
- Gradient boosting decision trees
- Captures non-linear relationships
- Handles feature interactions
- Often competitive with neural networks on tabular data

### 3. Attention Model
- Multi-head self-attention architecture
- 3 layers, 8 attention heads, 512 hidden dim
- Batch normalization + dropout (0.2)
- Strong regularization (weight decay: 0.05)
- Best for complex temporal patterns

### 4. Ensemble Model
- Combines Linear, XGBoost, and Attention
- Weighted averaging with optimized weights
- Weights optimized on validation set
- Reduces variance through diversification

## Critical Bias Fixes & Anti-Overfitting Measures

### ⚠️ IMPORTANT: Overfitting Prevention

This system includes multiple safeguards against overfitting and bias:

**1. Look-Ahead Bias Prevention**
- All features lagged by 1 bar via `FeatureLagger` (src/features/feature_lagger.py:1)
- Features at time T only use data up to T-1
- Verified by automated tests (tests/test_features.py:1)

**2. Circular Logic Prevention**
- Backtest returns calculated from actual entry/exit prices
- Does NOT use `forward_return` column (training label) for P&L
- Uses `RealReturnCalculator` (src/backtesting/return_calculator.py:1)
- Verified by automated tests (tests/test_backtest.py:1)

**3. Walk-Forward Validation**
- Expanding window validation for temporal stability
- Initial train: 1000 samples, test window: 200, step: 100
- Tests performance across multiple out-of-sample periods
- Detects model degradation over time

**4. Regularization**
- **Linear**: Ridge L2 regularization (alpha=1.0)
- **XGBoost**: Early stopping (10 rounds), max depth=5
- **Attention**: Dropout (0.2), weight decay (0.05), batch norm, early stopping (10 epochs)
- **Ensemble**: Diversification across model types

**5. Realistic Cost Assumptions**
- 8 bps per trade (commission + slippage)
- No fee-free trading assumptions
- Accounts for market impact

### Expected Performance

**Realistic Expectations (After Fixes):**
- Total Return: 5-15% annually
- Sharpe Ratio: 1.5-2.5
- Win Rate: 50-55%
- Max Drawdown: -20% to -35%
- Transaction cost drag: -2% to -5% annually

Note: Early versions without bias fixes showed inflated returns (~34%, Sharpe 7.7) that were not achievable in live trading.

## Configuration

All parameters centralized in `config/settings.py`:

```python
# Trading Parameters
TICKERS = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT', 'GLD']
HOLDING_PERIOD_MINUTES = 5
INITIAL_CAPITAL = 100000

# Transaction Costs
COMMISSION_BPS = 5.0  # 5 basis points
SLIPPAGE_BPS = 3.0    # 3 basis points

# Data Paths
FEATURES_PARQUET = "data/trump-truth-social-archive/data/truth_archive_with_features.parquet"

# Model Hyperparameters
ATTENTION_HIDDEN_DIM = 512
ATTENTION_NUM_HEADS = 8
ATTENTION_WEIGHT_DECAY = 0.05

LINEAR_N_FEATURES = 50
LINEAR_ALPHA = 1.0

XGBOOST_MAX_DEPTH = 5
XGBOOST_N_ESTIMATORS = 100
```

## Running Backtests

### Basic Single Model Backtest

```python
from src.models.linear_model import LinearModel
from src.backtesting.backtest_engine import BacktestEngine

# Create model
model = LinearModel(name="LinearBaseline", n_features=50, alpha=1.0)

# Create backtest engine with transaction costs
engine = BacktestEngine(
    model=model,
    tickers=['SPY'],
    apply_transaction_costs=True  # 8 bps per trade
)

# Run backtest
results = engine.run_single_ticker('SPY')
```

### Compare Multiple Models

```python
from src.models.linear_model import LinearModel
from src.models.xgboost_model import XGBoostModel
from src.models.attention_model import AttentionModel

models = [
    LinearModel(name="Linear"),
    XGBoostModel(name="XGBoost"),
    AttentionModel(name="Attention")
]

for model in models:
    engine = BacktestEngine(model=model, tickers=['SPY'])
    results = engine.run_single_ticker('SPY')
```

Results saved to: `output/{model_name}/{ticker}/`

### Walk-Forward Validation

```python
from src.backtesting.walk_forward import WalkForwardValidator

validator = WalkForwardValidator(
    initial_train_size=1000,
    test_window_size=200,
    step_size=100
)

wf_results = validator.run_walk_forward(
    model=model,
    X=X, y=y, data_df=data_df,
    ticker='SPY',
    backtest_engine=engine
)
```

## Output

Backtest results are organized by model:

```
output/
├── LinearRegression/
│   └── SPY/
│       ├── SPY_trades.csv          # All trades with entry/exit times, returns
│       ├── SPY_equity_curve.csv    # Capital over time
│       └── SPY_metrics.txt         # Performance summary
├── XGBoost/
├── AttentionModel/
└── Ensemble/
```

### Metrics Calculated

- **Total Return**: (Final Capital - Initial Capital) / Initial Capital
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Mean P&L for winning vs losing trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Processing Time**: Per-tweet inference latency (mean, p95, p99)

## Testing

Comprehensive test suite with 32 tests:

```bash
cd tests
python run_all_tests.py
```

**Critical Tests:**
- Look-ahead bias detection (features don't use future data)
- Return calculation accuracy (no circular logic)
- Transaction cost calculations
- Model interface compliance
- Overfitting detection

See `IMPLEMENTATION_SUMMARY.md` for full details on architecture, bias fixes, and testing.

---

# Part 2: L1 Market Data Analysis

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
