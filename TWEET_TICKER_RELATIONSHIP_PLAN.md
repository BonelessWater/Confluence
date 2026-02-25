# Plan: Discovering Tweet-Ticker Relationships and Influence Scoring

## Overview

This document outlines multiple approaches to systematically discover relationships between Trump's tweets and available stock tickers, and to rate the influence/volatility impact of each tweet on each ticker using NLP, bag-of-words, and LLM-based methods.

## Current State

### Available Data
- **Tweets**: Stored in `data/trump-truth-social-archive/data/truth_archive_with_features.parquet`
  - Tweet content, timestamps, embeddings (likely transformer-based)
  - Engagement metrics (replies, reblogs, favourites)
  - PCA-reduced embeddings (50 components) + original embeddings

- **Tickers**: Multiple asset classes available
  - **US Equity**: SPY, QQQ, DIA, IWM, AMD, NVDA, QCOM
  - **EM Equity**: EWW (Mexico ETF)
  - **FX**: CYB (China Yuan), UUP (US Dollar)
  - **Commodities**: GLD (Gold), USO (Oil)
  - **Fixed Income**: TLT, IEF, SHY (Treasuries)

- **Market Data**: 1-minute OHLCV bars in parquet format
- **Forward Returns**: Calculated at 5, 15, 30, 60 minute horizons

### Current Limitations
- Manual keyword-to-ticker mapping (hardcoded signals)
- No systematic discovery of relationships
- No per-tweet, per-ticker influence scoring
- Embeddings exist but not leveraged for ticker-specific relationships

---

## Option 1: Statistical Correlation Discovery (Baseline)

### Approach
Use statistical methods to discover relationships between tweet features and ticker returns.

### Methodology

#### 1.1 Feature-Ticker Correlation Matrix
- **Input**: Tweet embeddings (PCA components), bag-of-words features, engagement metrics
- **Output**: Correlation matrix between each feature and each ticker's forward returns
- **Implementation**:
  ```python
  # For each ticker:
  correlations = {}
  for ticker in TICKERS:
      for feature_col in embedding_cols + bag_of_words_cols:
          corr = df[feature_col].corr(df[f'{ticker}_30m'])
          correlations[(ticker, feature_col)] = corr
  ```

#### 1.2 Mutual Information Scoring
- Measure non-linear relationships using mutual information
- More robust than correlation for non-linear patterns
- **Library**: `sklearn.feature_selection.mutual_info_regression`

#### 1.3 Granger Causality Testing
- Test if tweet features predict ticker returns (beyond autocorrelation)
- **Library**: `statsmodels.tsa.stattools.grangercausalitytests`
- **Output**: p-values indicating predictive power

### Pros
- Fast computation
- Interpretable results
- No external dependencies
- Works with existing embeddings

### Cons
- Assumes linear/near-linear relationships
- May miss complex semantic relationships
- Requires large sample sizes for significance

### Output Format
```python
{
    'ticker': 'SPY',
    'feature': 'embedding_pca_12',
    'correlation': 0.23,
    'p_value': 0.001,
    'mutual_info': 0.15,
    'granger_p_value': 0.02
}
```

---

## Option 2: Bag-of-Words with Ticker-Specific Scoring

### Approach
Extract keywords/n-grams from tweets and score their relationship to each ticker.

### Methodology

#### 2.1 Keyword Extraction
- **Unigrams**: Single words (e.g., "tariff", "china", "trade")
- **Bigrams**: Two-word phrases (e.g., "trade war", "stock market")
- **Trigrams**: Three-word phrases (e.g., "make america great")
- **TF-IDF**: Term frequency-inverse document frequency weighting
- **Library**: `sklearn.feature_extraction.text.TfidfVectorizer`

#### 2.2 Ticker-Specific Keyword Scoring
For each ticker, calculate:
- **Mean Return**: Average forward return when keyword appears
- **T-Statistic**: Statistical significance of mean return
- **Hit Rate**: Percentage of positive returns
- **Volatility Impact**: Change in realized volatility after tweet

```python
def score_keyword_ticker_relationship(keyword, ticker, tweets_df, returns_df):
    # Find tweets containing keyword
    keyword_tweets = tweets_df[tweets_df['content'].str.contains(keyword, case=False)]
    
    # Get corresponding returns
    keyword_returns = returns_df.loc[keyword_tweets.index, f'{ticker}_30m']
    
    # Calculate metrics
    mean_return = keyword_returns.mean()
    t_stat = stats.ttest_1samp(keyword_returns, 0).statistic
    hit_rate = (keyword_returns > 0).mean()
    vol_impact = keyword_returns.std() - returns_df[f'{ticker}_30m'].std()
    
    return {
        'keyword': keyword,
        'ticker': ticker,
        'mean_return': mean_return,
        't_stat': t_stat,
        'hit_rate': hit_rate,
        'vol_impact': vol_impact,
        'n_occurrences': len(keyword_tweets)
    }
```

#### 2.3 Per-Tweet Scoring
For each tweet-ticker pair:
```python
def score_tweet_ticker_influence(tweet_text, ticker, keyword_scores):
    # Extract keywords from tweet
    keywords = extract_keywords(tweet_text)
    
    # Aggregate scores
    influence_score = sum(
        keyword_scores.get((kw, ticker), {}).get('mean_return', 0)
        for kw in keywords
    )
    
    volatility_score = sum(
        keyword_scores.get((kw, ticker), {}).get('vol_impact', 0)
        for kw in keywords
    )
    
    return {
        'influence_score': influence_score,
        'volatility_score': volatility_score,
        'matched_keywords': keywords
    }
```

### Pros
- Interpretable (shows which keywords matter)
- Fast computation
- Can handle large vocabularies
- Works with existing infrastructure

### Cons
- Loses word order and context
- May miss semantic relationships (synonyms, paraphrases)
- Requires manual keyword curation or large n-gram sets

### Output Format
```python
{
    'tweet_id': '12345',
    'ticker': 'SPY',
    'influence_score': 0.0012,  # Expected return impact
    'volatility_score': 0.0005,  # Volatility increase
    'matched_keywords': ['china', 'trade'],
    'keyword_contributions': {
        'china': 0.0007,
        'trade': 0.0005
    }
}
```

---

## Option 3: Embedding-Based Similarity Matching

### Approach
Use existing tweet embeddings to find similar historical tweets and infer ticker relationships.

### Methodology

#### 3.1 Historical Tweet Clustering
- Cluster tweets by embedding similarity (e.g., K-means, DBSCAN)
- For each cluster, calculate average returns per ticker
- **Library**: `sklearn.cluster.KMeans`, `faiss` for fast similarity search

#### 3.2 Similarity-Based Scoring
For a new tweet:
```python
def score_by_similarity(new_tweet_embedding, ticker, historical_data):
    # Find k most similar historical tweets
    similar_tweets = find_k_nearest(new_tweet_embedding, k=50)
    
    # Get returns for those tweets
    similar_returns = historical_data.loc[similar_tweets.index, f'{ticker}_30m']
    
    # Weight by similarity
    weights = cosine_similarity([new_tweet_embedding], similar_tweets.embeddings)[0]
    weighted_return = np.average(similar_returns, weights=weights)
    
    return {
        'expected_return': weighted_return,
        'confidence': weights.max(),  # How similar is the most similar tweet?
        'n_similar': len(similar_tweets)
    }
```

#### 3.3 Embedding-Ticker Regression
Train a regression model per ticker:
```python
# For each ticker:
X = tweet_embeddings  # Shape: (n_tweets, embedding_dim)
y = forward_returns[ticker]  # Shape: (n_tweets,)

model = Ridge(alpha=1.0)
model.fit(X, y)

# For new tweet:
influence_score = model.predict([new_tweet_embedding])[0]
```

### Pros
- Leverages existing embeddings
- Captures semantic relationships
- Can generalize to unseen keywords
- Fast inference once trained

### Cons
- Requires good-quality embeddings
- May overfit if embedding space is noisy
- Less interpretable than keyword-based methods

### Output Format
```python
{
    'tweet_id': '12345',
    'ticker': 'SPY',
    'influence_score': 0.0015,  # From regression model
    'similarity_score': 0.87,   # Max similarity to historical tweets
    'top_similar_tweets': [tweet_id1, tweet_id2, ...],
    'confidence': 0.85
}
```

---

## Option 4: LLM-Based Semantic Analysis

### Approach
Use Large Language Models to explicitly analyze tweet content and rate ticker-specific influence.

### Methodology

#### 4.1 Prompt-Based Scoring
For each tweet-ticker pair, use LLM to score:
```python
def llm_score_tweet_ticker(tweet_text, ticker, ticker_description):
    prompt = f"""
    Analyze this tweet and rate its potential impact on {ticker} ({ticker_description}).
    
    Tweet: "{tweet_text}"
    
    Provide:
    1. Influence score (-1 to +1): Expected return impact
    2. Volatility score (0 to 1): Expected volatility increase
    3. Confidence (0 to 1): How certain is this assessment?
    4. Reasoning: Brief explanation
    
    Format as JSON:
    {{
        "influence_score": <float>,
        "volatility_score": <float>,
        "confidence": <float>,
        "reasoning": "<text>"
    }}
    """
    
    response = llm.generate(prompt)
    return parse_json(response)
```

#### 4.2 Batch Processing with Embeddings
- Generate embeddings for all tweets
- Use LLM to analyze representative tweets from each cluster
- Propagate scores to similar tweets

#### 4.3 Fine-Tuned Model
- Fine-tune a smaller model (e.g., DistilBERT, RoBERTa) on historical tweet-ticker-return pairs
- Input: Tweet text + ticker name
- Output: Influence score, volatility score
- **Training Data**: Historical tweets with actual forward returns

```python
# Training example
{
    'input': 'Tweet: "China trade deal is great!" | Ticker: SPY',
    'output': {
        'influence_score': 0.0007,  # Actual forward return
        'volatility_score': 0.0003   # Actual volatility change
    }
}
```

### Pros
- Captures complex semantic relationships
- Can understand context and nuance
- Handles synonyms and paraphrases naturally
- Can incorporate domain knowledge (e.g., "tariff" → EWW)

### Cons
- **Cost**: API calls can be expensive at scale
- **Latency**: Slower than statistical methods
- **Reproducibility**: Non-deterministic outputs
- **Bias**: May inherit LLM biases

### LLM Options
1. **OpenAI GPT-4/GPT-3.5**: High quality, paid API
2. **Anthropic Claude**: Good reasoning, paid API
3. **Open Source**: Llama 2/3, Mistral (self-hosted, free but requires GPU)
4. **Specialized**: FinBERT (financial domain pre-trained)

### Output Format
```python
{
    'tweet_id': '12345',
    'ticker': 'SPY',
    'influence_score': 0.0008,
    'volatility_score': 0.0004,
    'confidence': 0.75,
    'reasoning': 'Tweet mentions trade positively, historically bullish for SPY',
    'method': 'llm_gpt4'
}
```

---

## Option 5: Hybrid Ensemble Approach

### Approach
Combine multiple methods and weight by their historical performance.

### Methodology

#### 5.1 Multi-Method Scoring
For each tweet-ticker pair, generate scores from:
- Statistical correlation (Option 1)
- Bag-of-words (Option 2)
- Embedding similarity (Option 3)
- LLM analysis (Option 4)

#### 5.2 Weighted Ensemble
```python
def ensemble_score(tweet, ticker, methods=['correlation', 'bow', 'embedding', 'llm']):
    scores = {}
    
    if 'correlation' in methods:
        scores['correlation'] = correlation_score(tweet, ticker)
    if 'bow' in methods:
        scores['bow'] = bag_of_words_score(tweet, ticker)
    if 'embedding' in methods:
        scores['embedding'] = embedding_score(tweet, ticker)
    if 'llm' in methods:
        scores['llm'] = llm_score(tweet, ticker)
    
    # Weight by historical accuracy
    weights = {
        'correlation': 0.2,
        'bow': 0.3,
        'embedding': 0.3,
        'llm': 0.2
    }
    
    final_score = sum(
        scores[method] * weights[method]
        for method in scores.keys()
    )
    
    return {
        'final_influence_score': final_score,
        'method_scores': scores,
        'method_weights': weights
    }
```

#### 5.3 Dynamic Weighting
- Train a meta-model to predict which method works best for each tweet-ticker pair
- Use features: tweet length, engagement, time of day, ticker volatility, etc.

### Pros
- Combines strengths of all methods
- More robust than single method
- Can adapt to different tweet types

### Cons
- More complex to implement
- Requires tuning weights
- Slower (runs multiple methods)

---

## Recommended Implementation Plan

### Phase 1: Baseline Discovery (Week 1)
**Goal**: Establish baseline relationships using existing data

1. **Statistical Correlation** (Option 1)
   - Compute correlation matrix: embeddings × ticker returns
   - Identify top features per ticker
   - Output: `ticker_feature_correlations.csv`

2. **Bag-of-Words Discovery** (Option 2)
   - Extract top keywords/n-grams
   - Score each keyword-ticker pair
   - Output: `keyword_ticker_scores.csv`

**Deliverables**:
- Correlation heatmaps per ticker
- Top 20 keyword-ticker relationships
- Baseline influence scoring function

### Phase 2: Embedding Enhancement (Week 2)
**Goal**: Leverage existing embeddings for better relationships

1. **Embedding Regression** (Option 3)
   - Train Ridge regression per ticker: `embedding → forward_return`
   - Cross-validate to prevent overfitting
   - Output: Trained models + influence scores

2. **Similarity-Based Scoring**
   - Implement k-NN similarity search
   - Score new tweets by historical similarity
   - Output: Similarity-based influence scores

**Deliverables**:
- Trained embedding models (one per ticker)
- Similarity scoring function
- Comparison: Embedding vs. keyword methods

### Phase 3: LLM Integration (Week 3)
**Goal**: Add semantic understanding for complex relationships

1. **LLM Prototype** (Option 4)
   - Implement prompt-based scoring for sample tweets
   - Compare LLM scores vs. actual returns
   - Evaluate cost/latency trade-offs

2. **Fine-Tuning Option** (if cost-effective)
   - Prepare training dataset
   - Fine-tune smaller model (e.g., DistilBERT)
   - Evaluate vs. zero-shot LLM

**Deliverables**:
- LLM scoring function
- Cost/accuracy analysis
- Decision: Use LLM or fine-tuned model

### Phase 4: Ensemble & Production (Week 4)
**Goal**: Combine methods and create production system

1. **Ensemble Implementation** (Option 5)
   - Combine all methods with learned weights
   - Validate on held-out test set
   - Optimize for accuracy vs. latency

2. **Production System**
   - Create API/service for real-time scoring
   - Batch processing for historical analysis
   - Monitoring and evaluation framework

**Deliverables**:
- Ensemble scoring system
- Real-time API
- Performance dashboard

---

## Implementation Details

### Data Pipeline

```python
# 1. Load data
tweets_df = pd.read_parquet('data/.../truth_archive_with_features.parquet')
market_data = {ticker: pl.read_parquet(f'data/{ticker}.parquet') for ticker in TICKERS}

# 2. Calculate forward returns
returns_df = calculate_forward_returns(tweets_df, market_data, horizons=[5, 15, 30, 60])

# 3. Extract features
embeddings = extract_embeddings(tweets_df)
keywords = extract_keywords(tweets_df['tweet_content'])

# 4. Score relationships
for ticker in TICKERS:
    correlation_scores = compute_correlations(embeddings, returns_df[f'{ticker}_30m'])
    keyword_scores = score_keywords(keywords, returns_df[f'{ticker}_30m'])
    embedding_scores = train_embedding_model(embeddings, returns_df[f'{ticker}_30m'])
    
# 5. Generate per-tweet, per-ticker scores
influence_matrix = score_all_tweets_tickers(tweets_df, tickers, methods=['correlation', 'bow', 'embedding'])
```

### Output Schema

**Per-Tweet-Per-Ticker Scores**:
```python
{
    'tweet_id': str,
    'ticker': str,
    'influence_score': float,        # Expected return impact (-1 to +1)
    'volatility_score': float,       # Expected volatility increase (0 to 1)
    'confidence': float,             # Confidence in score (0 to 1)
    'method': str,                   # 'correlation', 'bow', 'embedding', 'llm', 'ensemble'
    'method_scores': dict,           # Individual method scores
    'matched_keywords': list,        # Keywords that triggered scoring
    'historical_similarity': float,  # Similarity to historical tweets
    'timestamp': datetime
}
```

**Aggregate Ticker Relationships**:
```python
{
    'ticker': str,
    'top_keywords': [{'keyword': str, 'mean_return': float, 't_stat': float}],
    'top_embedding_features': [{'feature': str, 'correlation': float}],
    'avg_influence': float,
    'avg_volatility_impact': float,
    'n_tweets_analyzed': int
}
```

### Evaluation Metrics

1. **Prediction Accuracy**: Correlation between predicted influence scores and actual forward returns
2. **Ranking Quality**: Can we rank tweets by actual impact? (NDCG, Spearman correlation)
3. **Volatility Prediction**: How well do volatility scores predict realized volatility?
4. **Trading Performance**: If we trade on top-scored tweets, what's the Sharpe ratio?

---

## Cost & Resource Estimates

### Option 1: Statistical Correlation
- **Time**: 1-2 hours
- **Cost**: $0 (uses existing data)
- **Compute**: Single CPU, minimal memory

### Option 2: Bag-of-Words
- **Time**: 2-4 hours
- **Cost**: $0
- **Compute**: Single CPU, moderate memory for large vocabularies

### Option 3: Embedding-Based
- **Time**: 4-8 hours (including model training)
- **Cost**: $0
- **Compute**: Single CPU, moderate memory

### Option 4: LLM-Based
- **Time**: 8-16 hours (including prompt engineering)
- **Cost**: 
  - GPT-4: ~$0.03 per tweet (if 1000 tweets × 14 tickers = $420)
  - GPT-3.5: ~$0.001 per tweet ($14 total)
  - Self-hosted (Llama): $0 but requires GPU
- **Compute**: API calls or GPU for self-hosted

### Option 5: Ensemble
- **Time**: 12-20 hours
- **Cost**: Depends on LLM usage
- **Compute**: Moderate

---

## Recommendations

### For Quick Start (1-2 days)
1. Implement **Option 2 (Bag-of-Words)** first
   - Fast to implement
   - Interpretable results
   - Good baseline

2. Add **Option 3 (Embedding Regression)**
   - Leverages existing embeddings
   - Better generalization than keywords

### For Best Accuracy (1-2 weeks)
1. Implement all options (1-4)
2. Compare performance on validation set
3. Build **Option 5 (Ensemble)** with learned weights
4. Use LLM selectively (e.g., only for high-confidence tweets or ambiguous cases)

### For Production System
1. Start with **Option 2 + Option 3** (fast, interpretable)
2. Add **Option 4 (LLM)** for edge cases or high-value tweets
3. Use **Option 5 (Ensemble)** to combine methods
4. Monitor performance and retrain periodically

---

## Next Steps

1. **Choose primary method(s)** based on accuracy vs. cost trade-offs
2. **Implement data pipeline** for loading tweets, market data, calculating returns
3. **Build scoring functions** for selected method(s)
4. **Evaluate on historical data** and compare to actual returns
5. **Create visualization dashboard** showing tweet-ticker relationships
6. **Deploy scoring API** for real-time or batch processing

---

## Questions to Resolve

1. **LLM Budget**: What's the acceptable cost for LLM-based scoring?
2. **Latency Requirements**: Real-time (<1s) or batch processing acceptable?
3. **Interpretability**: Do we need explainable scores (keyword-based) or is black-box OK?
4. **Scope**: Score all tweet-ticker pairs or only top candidates?
5. **Update Frequency**: How often should relationships be re-discovered? (Daily, weekly, monthly?)

---

**Last Updated**: February 8, 2026
**Status**: Planning Phase
