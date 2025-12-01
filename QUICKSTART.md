# Quick Start Guide

## Running Your First Analysis

### 1. Basic Analysis (All Stocks)

```bash
cd src
python analyze.py --parquet ../data/market_data.parquet
```

### 2. Analyze Specific Stocks

```bash
python analyze.py --parquet ../data/market_data.parquet --tickers AAPL,MSFT,GOOGL
```

### 3. Limit Rows for Faster Processing

```bash
python analyze.py --parquet ../data/market_data.parquet --max-rows 25000
```

### 4. Custom Output Directory

```bash
python analyze.py --parquet ../data/market_data.parquet --output /path/to/results
```

## Understanding the Output

### Console Output
The script prints:
- Status updates for each stock processed
- Per-stock performance ranking
- Cross-stock method comparison
- Overall method ranking (0-10 scale)

### Files Generated

**l1_multi_stock_summary.csv**
- Summary of all predictions
- Columns: Stock, Method, MAE, RMSE, MAPE, R-squared, Correlation, Time, Predictions

**l1_multi_stock_comparison.png**
4-panel visualization:
1. MAE by stock and method
2. R-squared by stock and method
3. Average performance across stocks
4. Computation time comparison

## Interpreting Results

### Best Method Selection

Choose a method based on your needs:

- **Consensus Mid**: Fastest, good for real-time needs (low latency)
- **Consensus Micro**: Moderate speed, better accuracy than Mid
- **EMA Estimate**: Fast and smooth, good for trend following
- **Kalman Filter**: Best accuracy, slightly slower

### Ranking Score (0-10)

Higher is better. Score based on:
- 50% accuracy (lower MAE)
- 35% quality (higher R²)
- 15% speed (lower computation time)

### Performance Metrics

**MAE** (Mean Absolute Error)
- Average price error in dollars
- Lower is better
- Use when you care about absolute accuracy

**R-squared** (0-1)
- Percentage of price variance explained
- Higher is better
- 0.7+ is excellent, 0.5+ is good

**Time (microseconds)**
- Computation per update
- Lower is better
- Kalman ~50x slower than Mid

## Common Use Cases

### 1. Real-Time Trading
Want the fastest method that's still accurate?
```bash
python analyze.py --parquet ../data/market_data.parquet \
                  --max-rows 10000 \
                  --output ../output/realtime
```
Look for: EMA Estimate or Consensus Micro

### 2. Backtesting Strategy
Want the most accurate price estimate?
```bash
python analyze.py --parquet ../data/market_data.parquet \
                  --output ../output/backtest
```
Look for: Kalman Filter (highest R²)

### 3. Quick Comparison
Want to test a few specific stocks?
```bash
python analyze.py --parquet ../data/market_data.parquet \
                  --tickers AAPL,MSFT \
                  --max-rows 50000
```

### 4. Optimize for Your Exchange Mix
Run on stocks with specific exchange combinations:
```bash
python analyze.py --parquet ../data/market_data.parquet \
                  --tickers STOCK1,STOCK2,STOCK3
```

## Extending the Code

### Adding a New Method

1. Create a new file in `src/L1_transform/my_method.py`:
```python
class MyEfficientPrice:
    def __init__(self, param=0.1):
        self.param = param
        self.state = None

    def update(self, price):
        # Your estimation logic
        self.state = price * self.param
        return self.state
```

2. Add to `analyze.py` in the `generate_predictions()` function:
```python
from L1_transform.my_method import MyEfficientPrice

# In generate_predictions():
my_method = MyEfficientPrice()
my_val = my_method.update(mid)

# Store prediction:
results['my_method_pred'].append(my_val)
```

3. Add to metric calculations (same function).

### Using Methods Independently

```python
from L1_transform import (
    MultiExchangeBook,
    compute_consensus_mid,
    EMAEfficientPrice,
    PredictionMetrics
)

# Create order book
book = MultiExchangeBook()
book.update_quote('NYSE', 'BID', 100.0, 1000, timestamp)
book.update_quote('NYSE', 'ASK', 100.1, 1000, timestamp)
book.update_quote('NASDAQ', 'BID', 100.05, 1000, timestamp)
book.update_quote('NASDAQ', 'ASK', 100.15, 1000, timestamp)

# Get price estimate
mid_price = compute_consensus_mid(book, timestamp)

# Calculate metrics
metrics = PredictionMetrics.calculate_accuracy_metrics(actuals, predictions)
print(f"MAE: {metrics['mae']:.6f}")
```

## Troubleshooting

### Q: "No valid stocks could be processed"
**A**: Check that your parquet file has:
- A 'Ticker' column
- Multiple exchanges per stock
- Valid prices and quantities (> 0)

### Q: "ImportError: cannot import name..."
**A**: Make sure you're in the `src/` directory when running:
```bash
cd src
python analyze.py ...
```

### Q: Analysis is very slow
**A**: Use `--max-rows` to limit data:
```bash
python analyze.py --parquet ../data/market_data.parquet --max-rows 10000
```

### Q: Out of memory error
**A**: Process fewer rows or specific tickers:
```bash
python analyze.py --parquet ../data/market_data.parquet \
                  --tickers AAPL \
                  --max-rows 25000
```

## Next Steps

1. Review the README.md for detailed method documentation
2. Check the generated CSV and PNG outputs
3. Experiment with different `--max-rows` and `--tickers` combinations
4. Extend with your own consolidation methods
