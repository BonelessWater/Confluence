# Confluence — Trading Strategy Research

Event-driven trading pipeline that exploits social media signals from a high-profile political figure across 13 asset classes.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run all scoring methods and backtests
python scripts/run_all_tweet_ticker_methods.py

# Run a specific method
python scripts/run_all_tweet_ticker_methods.py --method correlation
python scripts/run_all_tweet_ticker_methods.py --method sentiment

# Run tests
python -m pytest tests/ -v
```

➡ For the full pipeline walkthrough see **[PIPELINE_README.md](PIPELINE_README.md)**.

---

## Asset Universe (13 Tickers)

| Ticker | Description | Category |
|--------|-------------|----------|
| SPY | S&P 500 ETF | US Equity |
| DIA | Dow Jones ETF | US Equity |
| AMD | Advanced Micro Devices | Tech |
| NVDA | NVIDIA | Tech |
| QCOM | Qualcomm | Tech |
| EWW | iShares MSCI Mexico ETF | EM Equity |
| CYB | China Yuan ETF | FX |
| UUP | US Dollar Index ETF | FX |
| GLD | Gold ETF | Commodity |
| USO | US Oil ETF | Commodity |
| TLT | 20+ Year Treasury ETF | Fixed Income |
| IEF | 7-10 Year Treasury ETF | Fixed Income |
| SHY | 1-3 Year Treasury ETF | Fixed Income |

---

## System Architecture

```
confluence/
├── config/settings.py                  # All parameters (tickers, costs, horizons)
├── src/
│   ├── tweet_ticker_analysis/          # Primary pipeline (Flow B)
│   │   ├── correlation_discovery.py   # Pearson + MI correlations → trades
│   │   ├── bag_of_words_scorer.py     # TF-IDF keyword → return signals
│   │   ├── embedding_scorer.py        # Embedding regression / k-NN
│   │   ├── sentiment_analyzer.py      # FinBERT + risk-off logic (all 13 tickers)
│   │   ├── ensemble_scorer.py         # Weighted combination of methods
│   │   ├── backtest_tweet_strategies.py  # Core backtester
│   │   ├── tweet_cleaner.py           # HTML strip, dedup, importance filtering
│   │   ├── correlation_validator.py   # t-test / p-value pre-trade validation
│   │   └── position_sizing.py         # Kelly Criterion / adaptive sizing
│   ├── models/                         # ML models (Flow A)
│   │   ├── linear_model.py            # Ridge regression baseline
│   │   ├── xgboost_model.py
│   │   ├── attention_model.py         # Multi-head attention (3L × 8H × 512d)
│   │   └── ensemble_model.py
│   ├── features/
│   │   ├── feature_engineer.py        # Price + tweet + interaction features
│   │   └── feature_lagger.py          # Lag by 1 bar — prevents look-ahead bias
│   └── backtesting/
│       ├── backtest_engine.py
│       ├── return_calculator.py       # Real entry/exit prices only
│       ├── transaction_costs.py       # 5 bps commission + 3 bps slippage
│       └── walk_forward.py
├── scripts/
│   ├── run_all_tweet_ticker_methods.py  # Main runner
│   ├── convert_tweet_csv_to_parquet.py
│   ├── precompute_features.py
│   └── improved_strategy_v3.py          # Event-driven keyword strategy (v3)
├── tests/
│   ├── test_backtest.py
│   ├── test_features.py
│   └── test_models.py
└── data/                               # Parquet files (tweets + OHLCV)
```

---

## Scoring Methods (Flow B)

| Method | Class | Speed | Signal Type |
|--------|-------|-------|-------------|
| Correlation | `CorrelationDiscovery` | Fast | Pearson + mutual information on embeddings/engagement |
| Bag-of-Words | `BagOfWordsScorer` | Fast | TF-IDF keywords with t-stat filtering |
| Embedding | `EmbeddingScorer` | Medium | Ridge regression / k-NN on tweet embeddings |
| Sentiment | `FinancialSentimentAnalyzer` | Medium | FinBERT + risk-off inversion for TLT/IEF/SHY/GLD |
| Ensemble | `EnsembleScorer` | Medium | Weighted combination of above methods |

All methods use a **70/30 temporal train/test split** to prevent look-ahead bias.

---

## Backtesting Parameters

| Parameter | Value |
|-----------|-------|
| Initial Capital | $100,000 |
| Commission | 5 bps |
| Slippage | 3 bps |
| **Round-trip cost** | **16 bps** |
| Default holding period | 30 min |
| Time filter | 12:00–14:00 ET |
| Max trades/day/ticker | 3 |
| Position sizing | Kelly Criterion (adaptive) |

### Trade Output Columns

Every `trades.csv` now includes:

| Column | Description |
|--------|-------------|
| `entry_time` / `exit_time` | Position timestamps |
| `duration_minutes` | Hold time in minutes |
| `ticker` | Asset traded |
| `direction` | `LONG` or `SHORT` |
| `position_size_usd` | Dollar value of position |
| `influence_score` | Raw model score |
| `signal_reason` | Scorer + top feature/keyword |
| `tweet_snippet` | First 120 chars of source tweet |
| `gross_return` / `net_return` | Before/after transaction costs |
| `pnl` | Profit & loss in dollars |

---

## Event-Driven Alpha Strategy (v3)

Keyword-asset pairs with statistically significant forward returns (t-stat > 2.0):

| Keyword | Asset | Direction | Holding | Avg Return | t-stat |
|---------|-------|-----------|---------|------------|--------|
| tariff | EWW | LONG | 60m | +0.32% | 3.06 |
| china | DIA | LONG | 30m | +0.11% | 2.59 |
| crash | TLT | LONG | 60m | +0.17% | 2.44 |
| weak | USO | SHORT | 30m | +0.15% | 2.79 |
| mexico | QCOM | SHORT | 15m | +0.12% | 2.85 |

**Out-of-sample results (v3):** +14.00% return | Sharpe 4.67 | 100 trades | 58% win rate

Statistical controls: Benjamini-Hochberg FDR correction, minimum t-stat > 2.0, 70/30 temporal split.

```bash
python scripts/improved_strategy_v3.py
```

---

## L1 Market Data Analysis

Evaluates four price consolidation methods against multi-exchange Level 1 data:

| Method | Description |
|--------|-------------|
| Consensus Mid | Simple midpoint across exchanges |
| Consensus Microprice | Volume-weighted microprice |
| EMA Estimate | Exponential moving average filter |
| Kalman Filter | Optimal linear state estimator |

```bash
cd src && python analyze.py --parquet ../data/market_data.parquet
```

Output: `output/l1_multi_stock_summary.csv`, `output/l1_multi_stock_comparison.png`

---

## Configuration

All parameters live in [`config/settings.py`](config/settings.py):

```python
TICKERS = ['SPY', 'DIA', 'AMD', 'NVDA', 'QCOM', 'EWW', 'CYB',
           'UUP', 'GLD', 'USO', 'TLT', 'IEF', 'SHY']
HOLDING_PERIOD_MINUTES = 30
INITIAL_CAPITAL = 100_000
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 3.0
```

---

## Statistical Appendix

**Multiple Testing (Benjamini-Hochberg FDR):**
For N tests at α = 0.05: order p-values, reject H₀ for all i ≤ k where pₖ ≤ (k/N) × FDR.

**t-statistic:** `t = x̄ / (s / √n)` — minimum |t| > 2.0 required for signal inclusion.

**Sharpe (annualized):** `(μ × √(252 × trades_per_day)) / σ`

---

*Disclaimer: Research purposes only. Past performance does not guarantee future results.*
