# Trading Strategy Backtesting & Market Data Analysis

This repository provides a comprehensive backtesting framework for trading strategies based on tweet sentiment analysis and machine learning models, along with tools for analyzing Level 1 (L1) market data consolidation methods.

## Overview

This repository contains three main systems:

1. **Event-Driven Alpha Strategy (v3)**: Asset-specific keyword signals with +14% out-of-sample returns
2. **Trading Strategy Backtesting System**: ML-based framework for tweet sentiment analysis
3. **L1 Market Data Analysis**: Tools for evaluating Level 1 market data consolidation methods

---

# Part 0: Event-Driven Alpha Strategy — Quantitative Research Note

*January 2026*

## Executive Summary

This research documents an event-driven trading strategy that exploits short-term price dislocations following social media posts from a high-profile political figure. Through rigorous statistical analysis of 13 asset classes, we identify keyword-asset pairs exhibiting statistically significant forward returns (t-stat > 2.0). The out-of-sample backtest yields **+14.00% returns** with a **Sharpe ratio of 4.67** over the test period.

Key findings:
- Asset-specific signals outperform broad market (SPY-only) approaches
- Optimal holding periods range from 15-60 minutes depending on signal type
- Time-of-day filtering (12:00-14:00 ET) significantly improves risk-adjusted returns
- Transaction costs of 8 bps are accounted for in all results

## 1. Introduction

### 1.1 Motivation

Social media posts from influential figures can trigger rapid, measurable price movements in financial markets. The challenge lies in separating genuine alpha from noise while controlling for multiple hypothesis testing bias.

### 1.2 Hypothesis

Specific keywords in social media posts exhibit heterogeneous effects across asset classes. A post mentioning "tariff" may have different implications for a Mexico ETF (EWW) than for broad US equities (SPY). By mapping keywords to their most responsive assets, we can extract alpha that a single-asset approach would miss.

## 2. Data

### 2.1 Signal Source

| Attribute | Value |
|-----------|-------|
| Source | Truth Social archive |
| Period | Full historical archive |
| Preprocessing | Timestamps aligned to market hours, duplicates removed |

### 2.2 Asset Universe

| Ticker | Description | Category |
|--------|-------------|----------|
| SPY | S&P 500 ETF | US Equity |
| DIA | Dow Jones ETF | US Equity |
| AMD | Advanced Micro Devices | Tech Equity |
| NVDA | NVIDIA Corporation | Tech Equity |
| QCOM | Qualcomm Inc | Tech Equity |
| EWW | iShares MSCI Mexico ETF | EM Equity |
| CYB | China Yuan ETF | FX |
| UUP | US Dollar Index ETF | FX |
| GLD | Gold ETF | Commodity |
| USO | US Oil ETF | Commodity |
| TLT | 20+ Year Treasury ETF | Fixed Income |
| IEF | 7-10 Year Treasury ETF | Fixed Income |
| SHY | 1-3 Year Treasury ETF | Fixed Income |

### 2.3 Market Data

- **Frequency:** 1-minute OHLCV bars
- **Source:** Databento (Parquet format)
- **Alignment:** Entry prices taken at minute following post timestamp

## 3. Methodology

### 3.1 Signal Construction

For each post, we:

1. Extract text content and normalize to lowercase
2. Scan for presence of target keywords
3. Map detected keywords to asset-specific trading signals
4. Apply time-of-day filter (12:00-14:00 ET)
5. Select strongest signal by t-statistic when multiple keywords present

### 3.2 Keyword-Asset Mapping

Signals are derived from historical analysis of keyword-return relationships:

| Keyword | Asset | Direction | Holding | Expected Return | t-stat |
|---------|-------|-----------|---------|-----------------|--------|
| tariff | EWW | LONG | 60m | +0.32% | 3.06 |
| china | DIA | LONG | 30m | +0.11% | 2.59 |
| china | SPY | LONG | 30m | +0.07% | 2.26 |
| crash | TLT | LONG | 60m | +0.17% | 2.44 |
| weak | USO | SHORT | 30m | +0.15% | 2.79 |
| mexico | QCOM | SHORT | 15m | +0.12% | 2.85 |
| great | QCOM | SHORT | 15m | +0.06% | 2.64 |

### 3.3 Combination Signals

When multiple keywords appear in a single post, combination signals take precedence:

| Keywords | Asset | Direction | Holding | Expected Return |
|----------|-------|-----------|---------|-----------------|
| tariff + china | SPY | LONG | 30m | +0.09% |
| tariff + trade | SPY | LONG | 30m | +0.125% |
| china + trade | SPY | LONG | 30m | +0.129% |

### 3.4 Statistical Validation

To control for data mining bias:

1. **Multiple Testing Correction:** Benjamini-Hochberg FDR applied to raw p-values
2. **t-statistic Threshold:** Minimum |t| > 2.0 required for signal inclusion
3. **Train/Test Split:** 70/30 temporal split with no lookahead
4. **Sample Size:** Minimum 5 observations per keyword-asset pair

## 4. Backtest Framework

### 4.1 Assumptions

| Parameter | Value |
|-----------|-------|
| Initial Capital | $100,000 |
| Commission | 5 bps |
| Slippage | 3 bps |
| **Total Cost** | **8 bps round-trip** |
| Max Trades/Day | 2 |
| Position Sizing | 100% of capital per trade |

### 4.2 Execution Model

- **Entry:** Close of minute following post timestamp
- **Exit:** Close of minute at holding period expiration
- **No Overlapping Positions:** Sequential trade execution only

### 4.3 Time Filter Rationale

Analysis of hourly return distributions revealed:

| Hour (ET) | Mean Return | Std Dev | Sharpe |
|-----------|-------------|---------|--------|
| 09:00 | -0.012% | 0.18% | -0.07 |
| 10:00 | +0.003% | 0.15% | +0.02 |
| 11:00 | +0.008% | 0.14% | +0.06 |
| **12:00** | **+0.025%** | **0.12%** | **+0.21** |
| **13:00** | **+0.036%** | **0.11%** | **+0.33** |
| **14:00** | **+0.018%** | **0.13%** | **+0.14** |
| 15:00 | +0.002% | 0.16% | +0.01 |

The 12:00-14:00 window exhibits both higher mean returns and lower volatility.

## 5. Results

### 5.1 Aggregate Performance

| Metric | Value |
|--------|-------|
| Total Return | +14.00% |
| Final Capital | $114,000.34 |
| Number of Trades | 100 |
| Win Rate | 58.0% |
| Average Return/Trade | +0.132% |
| Sharpe Ratio (annualized) | 4.67 |

### 5.2 Performance by Signal

| Signal | Trades | Win Rate | Avg Return |
|--------|--------|----------|------------|
| tariff → EWW | 11 | 100.0% | +0.499% |
| weak → USO | 5 | 60.0% | +0.305% |
| china+trade → SPY | 7 | 57.1% | +0.260% |
| china → DIA | 18 | 55.6% | +0.112% |
| china → SPY | 24 | 54.2% | +0.089% |
| crash → TLT | 3 | 66.7% | +0.156% |

### 5.3 Ablation Study: Time Filter Impact

| Configuration | Trades | Return |
|---------------|--------|--------|
| With filter (12-14) | 100 | +14.00% |
| Without filter | 187 | +7.84% |

The time filter reduces trade count by 47% while nearly doubling returns.

## 6. Risk Considerations

### 6.1 Limitations

1. **Sample Size:** Some signals have fewer than 20 observations
2. **Regime Dependence:** Strategy performance tied to political news cycle
3. **Capacity Constraints:** High-frequency nature limits scalable AUM
4. **Execution Risk:** 1-minute entry assumption may be optimistic
5. **Data Snooping:** Despite corrections, some overfit risk remains

### 6.2 Sensitivity Analysis

| Slippage (bps) | Return | Sharpe |
|----------------|--------|--------|
| 0 | +16.8% | 5.12 |
| 3 (base) | +14.0% | 4.67 |
| 5 | +12.3% | 4.21 |
| 10 | +8.4% | 3.14 |

### 6.3 Drawdown Profile

Maximum observed drawdown during test period: -2.3%

## 7. Quick Start

### Running the Strategy

```bash
# Run the improved v3 strategy backtest
python scripts/improved_strategy_v3.py

# Generate deep analysis report (signal discovery)
python scripts/deep_analysis_report.py

# Generate tweet-level HTML report
python scripts/comprehensive_tweet_report.py
```

### Key Files

| File | Purpose |
|------|---------|
| `scripts/improved_strategy_v3.py` | Main strategy implementation |
| `scripts/deep_analysis_report.py` | Multi-asset signal discovery |
| `scripts/comprehensive_tweet_report.py` | Interactive tweet/signal report |
| `output/improved_v3_trades.csv` | Trade log with P&L |

## 8. Conclusion

By decomposing social media signals into asset-specific components, we achieve meaningful alpha extraction where a naive single-asset approach fails. The tariff-EWW signal alone—exploiting the asymmetric impact of trade rhetoric on Mexican equities—delivers 100% hit rate across 11 trades.

The strategy's edge derives from:
1. **Cross-asset signal mapping** rather than SPY-only analysis
2. **Statistical filtering** to exclude noise keywords
3. **Time-of-day optimization** capturing midday liquidity dynamics
4. **Conservative position limits** to avoid overtrading

Future research directions include:
- Sentiment intensity scoring (beyond binary keyword detection)
- Volatility-adjusted position sizing
- Real-time deployment with latency optimization
- Expansion to options strategies for convexity

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

---

# Appendix: Statistical Methodology for Event-Driven Strategy

## A. Multiple Testing Correction

When testing N keyword-asset pairs, the probability of finding at least one false positive at significance level α is:

```
P(at least 1 false positive) = 1 - (1 - α)^N
```

For 100 tests at α = 0.05: P ≈ 99.4%

### Benjamini-Hochberg FDR Procedure

1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₙ
2. Find largest k where pₖ ≤ (k/N) × FDR
3. Reject H₀ for all i ≤ k

This controls the expected proportion of false discoveries rather than the family-wise error rate.

## B. t-statistic Calculation

For each keyword-asset pair with n observations:

```
t = (x̄ - μ₀) / (s / √n)

where:
  x̄ = sample mean return
  μ₀ = 0 (null hypothesis: no effect)
  s = sample standard deviation
  n = number of observations
```

Critical values (two-tailed, df ≈ ∞):
- |t| > 1.96 → p < 0.05
- |t| > 2.58 → p < 0.01
- |t| > 3.29 → p < 0.001

## C. Sharpe Ratio Calculation

```
Sharpe = (μ - rf) / σ × √(252 × trades_per_day)

where:
  μ = mean per-trade return
  rf = risk-free rate (assumed 0 for short holding periods)
  σ = standard deviation of per-trade returns
  252 = trading days per year
```

## D. Data Quality Filters

1. **Market Hours Only:** 09:30-16:00 ET
2. **Minimum Liquidity:** Trades only when bid-ask spread < 0.1%
3. **Price Continuity:** Forward-fill gaps up to 5 minutes
4. **Outlier Removal:** Returns > |5%| in 60 minutes excluded

---

*Disclaimer: This research is for educational purposes only. Past performance does not guarantee future results.*
