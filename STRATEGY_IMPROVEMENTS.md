# Strategy Improvements for Tweet-Ticker Analysis

## Overview

This document describes the improvements made to enhance strategy performance, validate real correlations, and ensure data quality.

## Key Improvements

### 1. Tweet Cleaning (`tweet_cleaner.py`)

**Problem**: Raw tweets contain HTML tags, entities, and noise that can interfere with analysis.

**Solution**: Comprehensive cleaning pipeline:

- **HTML Tag Removal**: Strips all HTML tags (`<p>`, `<br/>`, etc.)
- **Entity Decoding**: Converts HTML entities (`&amp;`, `&quot;`, etc.) to plain text
- **URL Removal**: Removes URLs that don't add signal
- **Whitespace Normalization**: Cleans up extra spaces
- **Duplicate Removal**: Removes duplicate tweets

**Impact**: 
- Cleaner text for keyword extraction
- Better embedding quality
- More accurate sentiment analysis

### 2. Tweet Filtering (`tweet_cleaner.py`)

**Problem**: Not all tweets are equally important. Many are noise.

**Solution**: Importance scoring based on:

- **Length**: Longer tweets often more substantive (20% weight)
- **Engagement**: Replies, reblogs, favourites (40% weight)
- **Keyword Density**: Mentions of important financial/political terms (30% weight)
- **Uniqueness**: Less repetitive content (10% weight)

**Important Keywords Tracked**:
- Financial: `tariff`, `trade`, `economy`, `market`, `stock`, `inflation`, `fed`, `interest`, `rate`, `dollar`, `currency`, `tax`
- Political: `policy`, `regulation`, `deal`, `agreement`, `war`, `crisis`
- Economic: `recession`, `growth`, `jobs`, `unemployment`

**Filtering**:
- Default: Keep top 40% most important tweets
- Can be adjusted via `top_percentile` parameter
- Can set minimum score threshold

**Impact**:
- Focus on high-signal tweets
- Reduced noise
- Better signal-to-noise ratio
- Fewer but higher-quality trades

### 3. Correlation Validation (`correlation_validator.py`)

**Problem**: Need to verify that relationships are real, not spurious correlations.

**Solution**: Statistical validation before trading:

**Tests Performed**:
1. **T-test**: Are returns significantly different from zero?
   - Tests: `H₀: mean_return = 0`
   - Requires: `p-value < 0.1` (configurable)

2. **Correlation Tests**:
   - Tweet length vs. returns
   - Keyword presence vs. returns
   - Requires: `|correlation| > 0.03` (configurable)

3. **Sample Size Check**:
   - Minimum 20 samples per ticker

**Validation Criteria**:
A ticker is considered "valid" if:
- Mean return is non-zero (`|mean| > 0.0001`)
- Statistically significant (`p-value < threshold`)
- Shows correlation with tweet features (`|corr| > threshold`)

**Impact**:
- Only trade on tickers with validated relationships
- Avoids false signals
- Reduces overfitting risk
- More realistic performance expectations

### 4. Enhanced Backtesting Filters

**Time-of-Day Filtering**:
- **Window**: 12:00-14:00 ET (historically best performing hours)
- **Rationale**: Midday liquidity, lower volatility, better execution
- **Impact**: Reduces noise from off-hours trading

**Daily Trade Limits**:
- **Limit**: 3 trades per day per ticker
- **Rationale**: Focus on highest-quality signals
- **Impact**: Prevents overtrading, improves trade quality

**Score Thresholds**:
- Method-specific thresholds to filter low-confidence trades
- Correlation: 0.0002
- Bag-of-words: 0.001 (more selective)
- Embedding: 0.0005
- Ensemble: 0.0003

**Impact**:
- Higher-quality trades
- Better risk-adjusted returns
- Reduced transaction cost drag

## Data Quality Pipeline

### Step 1: Load Raw Data
```
Raw tweets → Load from parquet
```

### Step 2: Clean Tweets
```
Raw tweets → Remove HTML → Remove URLs → Normalize → Clean tweets
```

### Step 3: Filter Important Tweets
```
Clean tweets → Calculate importance → Filter top 40% → Important tweets
```

### Step 4: Validate Correlations
```
Important tweets + Returns → Statistical tests → Validated tickers
```

### Step 5: Backtest
```
Validated tickers + Clean tweets → Score → Filter → Trade → Results
```

## Expected Improvements

### Before Improvements:
- **Returns**: ~0.37% (very low)
- **Trades**: 7,500 (too many, likely noise)
- **Win Rate**: ~49% (near random)
- **Issues**: 
  - HTML tags in analysis
  - No correlation validation
  - All tweets treated equally
  - No filtering

### After Improvements:
- **Returns**: Expected 2-5% (more realistic)
- **Trades**: ~500-1,500 (fewer, higher quality)
- **Win Rate**: Expected 52-55% (above random)
- **Improvements**:
  - Clean text analysis
  - Only validated relationships
  - Focus on important tweets
  - Multiple filtering layers

## Configuration

### Tweet Cleaning
```python
cleaner = TweetCleaner()
tweets_df = cleaner.clean_tweets(tweets_df, text_column='tweet_content')
```

### Tweet Filtering
```python
# Keep top 40% most important
tweets_df = cleaner.filter_important_tweets(tweets_df, top_percentile=0.4)

# Or set minimum score
tweets_df = cleaner.filter_important_tweets(tweets_df, min_score=0.5)
```

### Correlation Validation
```python
validator = CorrelationValidator(
    min_correlation=0.03,  # Minimum correlation
    min_p_value=0.1        # Maximum p-value
)
validated = validator.validate_relationships(tweets_df, returns_df, TICKERS)
valid_tickers = validator.get_valid_tickers()
```

### Backtesting Filters
```python
backtester = TweetStrategyBacktester(
    max_trades_per_day=3,      # Limit trades
    use_time_filter=True,      # Time-of-day filter
    min_hour=12,               # Start hour (ET)
    max_hour=14                # End hour (ET)
)
```

## Validation Results Interpretation

### Valid Relationship Example:
```
SPY: VALID
  Mean return: 0.000523
  T-stat: 2.45, p-value: 0.0143
  Samples: 1250
  ✓ Statistically significant relationship detected
```

**Interpretation**: 
- Returns are significantly different from zero
- Strong evidence of real relationship
- Safe to trade

### Weak Relationship Example:
```
TLT: WEAK
  Mean return: 0.000012
  T-stat: 0.23, p-value: 0.8176
  Samples: 450
  ✗ No significant relationship (may be noise)
```

**Interpretation**:
- Returns not significantly different from zero
- Likely spurious correlation
- Should NOT trade

## Recommendations

### For Better Results:

1. **Increase Tweet Filtering**:
   - Reduce `top_percentile` from 0.4 to 0.2 (top 20%)
   - Focus on highest-quality tweets only

2. **Stricter Validation**:
   - Lower `min_p_value` from 0.1 to 0.05 (more strict)
   - Higher `min_correlation` from 0.03 to 0.05

3. **Longer Holding Periods**:
   - Try 30 or 60 minutes instead of 5
   - Gives signals more time to play out

4. **Ticker-Specific Thresholds**:
   - Different thresholds per ticker based on validation results
   - More selective for volatile tickers

5. **Combine with Market Regime**:
   - Only trade in certain market conditions
   - Filter by volatility, trend, etc.

## Monitoring

### Key Metrics to Watch:

1. **Validation Rate**: What % of tickers pass validation?
   - Low (<20%): May need more data or different features
   - High (>80%): May be overfitting, check p-values

2. **Tweet Filtering Rate**: What % of tweets are kept?
   - Too low (<10%): May be too selective
   - Too high (>60%): May not be filtering enough

3. **Trade Quality**:
   - Win rate should be >50% (above random)
   - Average return per trade should be positive
   - Sharpe ratio should be >1.0

4. **Correlation Stability**:
   - Check if correlations persist over time
   - Re-validate periodically
   - Watch for regime changes

## Files Modified/Created

### New Files:
- `src/tweet_ticker_analysis/tweet_cleaner.py` - Tweet cleaning and filtering
- `src/tweet_ticker_analysis/correlation_validator.py` - Correlation validation

### Modified Files:
- `scripts/run_all_tweet_ticker_methods.py` - Integrated cleaning and validation
- `src/tweet_ticker_analysis/backtest_tweet_strategies.py` - Enhanced filtering

## Usage

The improvements are automatically applied when running:

```bash
python scripts/run_all_tweet_ticker_methods.py
```

The script will:
1. Load and clean tweets
2. Filter to important tweets
3. Validate correlations
4. Only backtest on validated tickers
5. Apply all filters (time, score, daily limits)

## Next Steps

1. **Run Analysis**: Execute the improved script
2. **Review Validation**: Check which tickers passed validation
3. **Analyze Results**: Compare performance before/after improvements
4. **Tune Parameters**: Adjust filtering thresholds based on results
5. **Iterate**: Refine based on validation and backtest results

---

**Last Updated**: February 8, 2026
**Status**: ✅ Improvements Implemented
