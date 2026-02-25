# Tweet-Ticker Relationship Analysis

This module implements multiple methods for discovering relationships between tweets and stock tickers, and scoring the influence/volatility impact of each tweet on each ticker.

## Methods Implemented

### 1. Correlation Discovery (`correlation_discovery.py`)
- **Approach**: Statistical correlation between tweet embeddings/features and ticker returns
- **Pros**: Fast, interpretable, no external dependencies
- **Cons**: Assumes linear relationships

### 2. Bag-of-Words Scorer (`bag_of_words_scorer.py`)
- **Approach**: Extract keywords/n-grams and score their relationship to each ticker
- **Pros**: Interpretable, fast, handles large vocabularies
- **Cons**: Loses context, misses semantic relationships

### 3. Embedding Scorer (`embedding_scorer.py`)
- **Approach**: Use existing embeddings for similarity matching and regression
- **Pros**: Leverages existing embeddings, captures semantics, generalizes well
- **Cons**: Requires good embeddings, less interpretable

### 4. LLM Scorer (`llm_scorer.py`)
- **Approach**: Use Large Language Models to analyze tweet content
- **Pros**: Captures complex semantics, handles context/nuance
- **Cons**: Cost (API calls), latency, requires API key

### 5. Ensemble Scorer (`ensemble_scorer.py`)
- **Approach**: Combines all methods with learned weights
- **Pros**: Most robust, combines strengths
- **Cons**: Most complex, slower

## Usage

### Running All Methods

```bash
python scripts/run_all_tweet_ticker_methods.py
```

This will:
1. Load tweet and market data
2. Train all 5 methods
3. Backtest each method as a trading strategy
4. Generate quantstats equity curves and performance reports
5. Save results organized by method in `output/tweet_ticker_methods/`

### Output Structure

```
output/tweet_ticker_methods/
├── correlation/
│   ├── trades.csv
│   ├── equity_curve.csv
│   ├── metrics.txt
│   ├── equity_curve_simple.png
│   └── quantstats_report.html (if quantstats installed)
├── bag_of_words/
│   └── ...
├── embedding/
│   └── ...
├── ensemble/
│   └── ...
└── method_comparison.csv
```

### Using Individual Methods

```python
from src.tweet_ticker_analysis import CorrelationDiscovery, BagOfWordsScorer

# Initialize scorer
scorer = CorrelationDiscovery()

# Discover relationships
results = scorer.discover_relationships(
    tweets_df=tweets_df,
    returns_df=returns_df,
    tickers=['SPY', 'QQQ'],
    return_horizon='30m'
)

# Score a tweet-ticker pair
score = scorer.score_tweet_ticker(tweet_row, 'SPY', '30m')
print(f"Influence score: {score['influence_score']}")
```

## Requirements

- pandas, numpy, scikit-learn (all methods)
- quantstats (for equity curve reports)
- openai (for LLM method, optional)

Install with:
```bash
pip install quantstats
```

For LLM method, set `OPENAI_API_KEY` environment variable.

## Configuration

Edit `config/settings.py` to configure:
- `TICKERS`: List of tickers to analyze
- `HOLDING_PERIOD_MINUTES`: Holding period for trades
- `INITIAL_CAPITAL`: Starting capital
- Transaction costs (COMMISSION_BPS, SLIPPAGE_BPS)

## Notes

- All methods are trained on historical data
- Backtesting uses realistic transaction costs (8 bps per trade)
- Results are saved with method names for easy comparison
- Quantstats reports provide detailed performance analysis
