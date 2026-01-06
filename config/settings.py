"""
Centralized configuration for trading backtest system.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models" / "saved_models"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "output"

# Parquet files
TWEET_DATA_DIR = DATA_DIR / "trump-truth-social-archive" / "data"
TWEET_PARQUET = TWEET_DATA_DIR / "truth_archive_with_embeddings.parquet"
FEATURES_PARQUET = TWEET_DATA_DIR / "truth_archive_with_features.parquet"
FEATURES_LAGGED_PARQUET = PROCESSED_DATA_DIR / "features_with_lags.parquet"

# Trading parameters
TICKERS = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT', 'GLD']
HOLDING_PERIOD_MINUTES = 5
INITIAL_CAPITAL = 100000

# Transaction costs
COMMISSION_BPS = 5.0  # 5 basis points
SLIPPAGE_BPS = 3.0    # 3 basis points
TOTAL_COST_BPS = COMMISSION_BPS + SLIPPAGE_BPS  # 8 bps per trade

# Model parameters
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Walk-forward validation parameters
WF_INITIAL_TRAIN_SIZE = 1000
WF_TEST_WINDOW_SIZE = 200
WF_STEP_SIZE = 100

# Feature engineering
FEATURE_LAG_PERIODS = 1  # Lag features by 1 bar to avoid look-ahead bias

# Random seed for reproducibility
RANDOM_SEED = 42

# Attention model hyperparameters
ATTENTION_HIDDEN_DIM = 512
ATTENTION_NUM_HEADS = 8
ATTENTION_NUM_LAYERS = 3
ATTENTION_DROPOUT = 0.2
ATTENTION_LEARNING_RATE = 0.001
ATTENTION_WEIGHT_DECAY = 0.05  # Increased from 0.01 for stronger regularization
ATTENTION_BATCH_SIZE = 64
ATTENTION_EPOCHS = 50
ATTENTION_EARLY_STOPPING_PATIENCE = 10

# Linear model hyperparameters
LINEAR_N_FEATURES = 50  # Select top 50 features
LINEAR_ALPHA = 1.0  # Ridge regularization

# XGBoost model hyperparameters
XGBOOST_MAX_DEPTH = 5
XGBOOST_N_ESTIMATORS = 100
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_EARLY_STOPPING_ROUNDS = 10

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
