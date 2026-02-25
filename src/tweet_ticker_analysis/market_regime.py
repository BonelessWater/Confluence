"""
Market regime detection and adaptive strategy adjustment.

Detects market conditions and adapts strategy parameters accordingly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime types."""
    LOW_VOL = "low_volatility"
    HIGH_VOL = "high_volatility"
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"


class MarketRegimeDetector:
    """
    Detect current market regime and adapt strategy parameters.
    """

    def __init__(self, 
                 vol_lookback: int = 20,
                 trend_lookback: int = 50,
                 vol_threshold_high: float = 0.02,
                 vol_threshold_low: float = 0.01):
        """
        Initialize regime detector.
        
        Args:
            vol_lookback: Days to look back for volatility calculation
            trend_lookback: Days to look back for trend detection
            vol_threshold_high: Volatility threshold for high vol regime
            vol_threshold_low: Volatility threshold for low vol regime
        """
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low

    def detect_regime(self, price_data: pd.DataFrame, ticker: str) -> MarketRegime:
        """
        Detect current market regime for a ticker.
        
        Args:
            price_data: DataFrame with OHLCV data
            ticker: Ticker symbol
            
        Returns:
            MarketRegime enum
        """
        if ticker not in price_data or len(price_data[ticker]) < self.trend_lookback:
            return MarketRegime.SIDEWAYS
        
        price_df = price_data[ticker]
        
        if 'close' not in price_df.columns:
            return MarketRegime.SIDEWAYS
        
        close_prices = price_df['close'].values
        
        # Calculate volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        recent_vol = np.std(returns[-self.vol_lookback:])
        
        # Calculate trend
        recent_prices = close_prices[-self.trend_lookback:]
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        trend_pct = trend_slope / recent_prices[0]
        
        # Determine regime
        if recent_vol > self.vol_threshold_high:
            vol_regime = MarketRegime.HIGH_VOL
        elif recent_vol < self.vol_threshold_low:
            vol_regime = MarketRegime.LOW_VOL
        else:
            vol_regime = MarketRegime.SIDEWAYS
        
        if trend_pct > 0.001:  # 0.1% per day = ~25% annual
            trend_regime = MarketRegime.TRENDING_UP
        elif trend_pct < -0.001:
            trend_regime = MarketRegime.TRENDING_DOWN
        else:
            trend_regime = MarketRegime.SIDEWAYS
        
        # Combine regimes (prioritize volatility)
        if vol_regime == MarketRegime.HIGH_VOL:
            return MarketRegime.HIGH_VOL
        elif vol_regime == MarketRegime.LOW_VOL:
            return MarketRegime.LOW_VOL
        else:
            return trend_regime

    def adapt_threshold(self, regime: MarketRegime, base_threshold: float) -> float:
        """
        Adapt score threshold based on regime.
        
        Args:
            regime: Current market regime
            base_threshold: Base threshold value
            
        Returns:
            Adapted threshold
        """
        multipliers = {
            MarketRegime.HIGH_VOL: 0.7,      # Lower threshold (more aggressive)
            MarketRegime.LOW_VOL: 1.5,       # Higher threshold (more selective)
            MarketRegime.BULL: 0.9,          # Slightly lower (bullish sentiment works)
            MarketRegime.BEAR: 1.2,          # Higher (be more selective)
            MarketRegime.TRENDING_UP: 0.85,
            MarketRegime.TRENDING_DOWN: 1.3,
            MarketRegime.SIDEWAYS: 1.0       # No change
        }
        
        multiplier = multipliers.get(regime, 1.0)
        return base_threshold * multiplier

    def adapt_position_size(self, regime: MarketRegime, base_size: float) -> float:
        """
        Adapt position size based on regime.
        
        Args:
            regime: Current market regime
            base_size: Base position size
            
        Returns:
            Adapted position size
        """
        multipliers = {
            MarketRegime.HIGH_VOL: 0.7,      # Smaller positions in high vol
            MarketRegime.LOW_VOL: 1.2,       # Larger positions in low vol
            MarketRegime.BULL: 1.1,
            MarketRegime.BEAR: 0.8,
            MarketRegime.TRENDING_UP: 1.05,
            MarketRegime.TRENDING_DOWN: 0.9,
            MarketRegime.SIDEWAYS: 1.0
        }
        
        multiplier = multipliers.get(regime, 1.0)
        return base_size * multiplier

    def get_regime_info(self, price_data: Dict[str, pd.DataFrame], 
                       tickers: List[str]) -> Dict[str, MarketRegime]:
        """
        Get regime for all tickers.
        
        Args:
            price_data: Dictionary of price DataFrames
            tickers: List of tickers
            
        Returns:
            Dictionary mapping ticker to regime
        """
        regimes = {}
        for ticker in tickers:
            if ticker in price_data:
                regimes[ticker] = self.detect_regime(price_data, ticker)
            else:
                regimes[ticker] = MarketRegime.SIDEWAYS
        
        return regimes
