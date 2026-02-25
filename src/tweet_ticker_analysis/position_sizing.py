"""
Advanced position sizing strategies.

Implements Kelly Criterion, volatility-adjusted sizing, and confidence-based sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PositionSizer:
    """
    Advanced position sizing using multiple strategies.
    """

    def __init__(self, 
                 max_position_size: float = 0.2,
                 use_kelly: bool = True,
                 kelly_fraction: float = 0.25):
        """
        Initialize position sizer.
        
        Args:
            max_position_size: Maximum position size as fraction of capital (0.2 = 20%)
            use_kelly: Whether to use Kelly Criterion
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, safer)
        """
        self.max_position_size = max_position_size
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self.trade_history = []  # Track trades for Kelly calculation

    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion optimal position size.
        
        Formula: f = (p * b - q) / b
        where:
        - p = win probability
        - q = loss probability (1 - p)
        - b = win/loss ratio (avg_win / avg_loss)
        
        Args:
            win_rate: Probability of winning (0 to 1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            
        Returns:
            Kelly fraction (0 to 1)
        """
        if avg_loss <= 0:
            return 0.1  # Default if no loss data
        
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        # Kelly formula
        kelly = (win_rate * b - q) / b
        
        # Apply fraction (quarter Kelly is common)
        kelly = kelly * self.kelly_fraction
        
        # Clamp between 0 and max
        kelly = max(0.0, min(kelly, self.max_position_size))
        
        return kelly

    def calculate_size(self,
                      influence_score: float,
                      confidence: float,
                      volatility: float,
                      win_rate: Optional[float] = None,
                      avg_win: Optional[float] = None,
                      avg_loss: Optional[float] = None) -> float:
        """
        Calculate optimal position size.
        
        Args:
            influence_score: Expected return from signal
            confidence: Confidence in signal (0 to 1)
            volatility: Current volatility (for risk adjustment)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average winning trade return (for Kelly)
            avg_loss: Average losing trade return (for Kelly)
            
        Returns:
            Position size as fraction of capital (0 to max_position_size)
        """
        # Base size from influence score (normalized)
        # Assume influence_score is in range -0.01 to +0.01 (typical tweet impact)
        base_size = abs(influence_score) * 100  # Scale to 0-1 range
        base_size = min(base_size, self.max_position_size)
        
        # Confidence adjustment
        confidence_adjustment = confidence
        adjusted_size = base_size * confidence_adjustment
        
        # Volatility adjustment (inverse relationship)
        # Higher volatility → smaller positions
        if volatility > 0:
            vol_adjustment = 1.0 / (1.0 + volatility * 10)
        else:
            vol_adjustment = 1.0
        
        adjusted_size = adjusted_size * vol_adjustment
        
        # Kelly Criterion adjustment (if data available)
        if self.use_kelly and win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_size = self.kelly_criterion(win_rate, avg_win, avg_loss)
            # Blend Kelly with signal-based size
            final_size = 0.6 * adjusted_size + 0.4 * kelly_size
        else:
            final_size = adjusted_size
        
        # Ensure within bounds
        final_size = max(0.01, min(final_size, self.max_position_size))
        
        return final_size

    def update_history(self, trade_result: Dict):
        """
        Update trade history for Kelly Criterion calculation.
        
        Args:
            trade_result: Dictionary with 'return' key
        """
        self.trade_history.append(trade_result)
        
        # Keep only recent history (last 100 trades)
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def get_statistics(self) -> Dict:
        """
        Get statistics from trade history for Kelly calculation.
        
        Returns:
            Dictionary with win_rate, avg_win, avg_loss
        """
        if len(self.trade_history) < 10:
            return {
                'win_rate': 0.5,
                'avg_win': 0.001,
                'avg_loss': 0.001
            }
        
        returns = [t['return'] for t in self.trade_history]
        returns = np.array(returns)
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
        avg_win = wins.mean() if len(wins) > 0 else 0.001
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }


class AdaptivePositionSizer(PositionSizer):
    """
    Position sizer that adapts based on recent performance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_performance = []
        self.performance_window = 20  # Last 20 trades

    def calculate_size(self, *args, **kwargs) -> float:
        """Calculate size with performance adaptation."""
        base_size = super().calculate_size(*args, **kwargs)
        
        # Adjust based on recent performance
        if len(self.recent_performance) >= 5:
            recent_win_rate = np.mean(self.recent_performance[-self.performance_window:])
            
            # If performing well, increase size slightly
            if recent_win_rate > 0.55:
                multiplier = 1.1
            # If performing poorly, decrease size
            elif recent_win_rate < 0.45:
                multiplier = 0.9
            else:
                multiplier = 1.0
            
            base_size = base_size * multiplier
        
        return base_size

    def update_performance(self, trade_result: Dict):
        """Update performance tracking."""
        super().update_history(trade_result)
        
        # Track win/loss (1 for win, 0 for loss)
        is_win = 1 if trade_result.get('return', 0) > 0 else 0
        self.recent_performance.append(is_win)
        
        # Keep only recent window
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance = self.recent_performance[-self.performance_window:]
