"""
Market regime detection for adaptive strategy adjustment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MarketRegimeDetector:
    """
    Detect market regime and adjust strategy accordingly.
    """

    def __init__(self, lookback_days: int = 20):
        """
        Initialize regime detector.
        
        Args:
            lookback_days: Days to look back for regime calculation
        """
        self.lookback_days = lookback_days

    def detect_regime(self, price_data: pd.DataFrame, 
                     ticker: Optional[str] = None) -> Dict:
        """
        Detect current market regime.
        
        Regimes:
        - 'bull': Strong uptrend (>2% in lookback)
        - 'bear': Strong downtrend (<-2% in lookback)
        - 'volatile': High volatility (>2% daily std)
        - 'calm': Low volatility (<1% daily std)
        - 'neutral': Sideways, moderate volatility
        
        Args:
            price_data: DataFrame with OHLCV data, indexed by timestamp
            ticker: Optional ticker name for logging
            
        Returns:
            Dictionary with regime info
        """
        if len(price_data) < self.lookback_days:
            return {
                'regime': 'neutral',
                'confidence': 0.0,
                'trend': 0.0,
                'volatility': 0.0
            }
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # Trend: Total return over lookback period
        if len(price_data) >= self.lookback_days:
            trend = (price_data['close'].iloc[-1] / price_data['close'].iloc[-self.lookback_days] - 1)
        else:
            trend = returns.mean() * self.lookback_days
        
        # Volatility: Rolling standard deviation
        volatility = returns.rolling(min(self.lookback_days, len(returns))).std().iloc[-1]
        
        # Determine regime
        if trend > 0.02:
            regime = 'bull'
            confidence = min(1.0, abs(trend) / 0.05)
        elif trend < -0.02:
            regime = 'bear'
            confidence = min(1.0, abs(trend) / 0.05)
        elif volatility > 0.02:
            regime = 'volatile'
            confidence = min(1.0, volatility / 0.04)
        elif volatility < 0.01:
            regime = 'calm'
            confidence = min(1.0, (0.01 - volatility) / 0.01)
        else:
            regime = 'neutral'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'trend': trend,
            'volatility': volatility,
            'lookback_days': self.lookback_days
        }

    def get_regime_adjustments(self, regime_info: Dict) -> Dict:
        """
        Get strategy adjustments for detected regime.
        
        Returns:
            Dictionary with adjustments:
            - threshold_multiplier: Adjust score threshold
            - position_size_multiplier: Adjust position size
            - min_confidence: Minimum confidence required
        """
        regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        adjustments = {
            'threshold_multiplier': 1.0,
            'position_size_multiplier': 1.0,
            'min_confidence': 0.0,
            'max_trades_per_day_multiplier': 1.0
        }
        
        if regime == 'bull':
            # Bull market: Can be more aggressive
            adjustments['threshold_multiplier'] = 0.8  # Lower threshold
            adjustments['position_size_multiplier'] = 1.1  # Slightly larger positions
            adjustments['min_confidence'] = 0.4
            adjustments['max_trades_per_day_multiplier'] = 1.2
        
        elif regime == 'bear':
            # Bear market: Be defensive
            adjustments['threshold_multiplier'] = 1.5  # Higher threshold
            adjustments['position_size_multiplier'] = 0.7  # Smaller positions
            adjustments['min_confidence'] = 0.6
            adjustments['max_trades_per_day_multiplier'] = 0.7
        
        elif regime == 'volatile':
            # High volatility: Be cautious
            adjustments['threshold_multiplier'] = 1.3
            adjustments['position_size_multiplier'] = 0.8
            adjustments['min_confidence'] = 0.5
            adjustments['max_trades_per_day_multiplier'] = 0.8
        
        elif regime == 'calm':
            # Low volatility: Can be more selective but confident
            adjustments['threshold_multiplier'] = 1.0
            adjustments['position_size_multiplier'] = 1.0
            adjustments['min_confidence'] = 0.5
            adjustments['max_trades_per_day_multiplier'] = 1.0
        
        else:  # neutral
            adjustments['threshold_multiplier'] = 1.0
            adjustments['position_size_multiplier'] = 1.0
            adjustments['min_confidence'] = 0.5
            adjustments['max_trades_per_day_multiplier'] = 1.0
        
        return adjustments

    def detect_regimes_for_tickers(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Detect regime for multiple tickers.
        
        Args:
            price_data: Dictionary of {ticker: price_df}
            
        Returns:
            Dictionary of {ticker: regime_info}
        """
        regimes = {}
        
        for ticker, price_df in price_data.items():
            try:
                regime_info = self.detect_regime(price_df, ticker)
                regimes[ticker] = regime_info
            except Exception as e:
                print(f"  Warning: Could not detect regime for {ticker}: {e}")
                regimes[ticker] = {
                    'regime': 'neutral',
                    'confidence': 0.0,
                    'trend': 0.0,
                    'volatility': 0.0
                }
        
        return regimes
