"""
Return calculator for backtesting.

CRITICAL: This module calculates actual realized returns from entry/exit prices,
NOT from the forward_return column which is a training label and should never be
used in backtesting PnL calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict

class RealReturnCalculator:
    """
    Calculate actual position returns from price data.

    This class fixes the circular backtesting logic where forward_return (a training label)
    was being used as the actual return. Instead, we calculate returns from entry and exit
    prices directly from the price data.
    """

    def __init__(self):
        """Initialize the return calculator."""
        self.price_cache = {}

    def load_price_data(self, price_df: pd.DataFrame, ticker: str):
        """
        Load and cache price data for a ticker.

        Args:
            price_df: DataFrame with OHLC price data indexed by timestamp
            ticker: Ticker symbol
        """
        self.price_cache[ticker] = price_df
        print(f"Loaded price data for {ticker}: {len(price_df)} bars")

    def calculate_return(self, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                        ticker: str, price_df: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate real return from entry to exit using actual prices.

        Args:
            entry_time: Position entry timestamp
            exit_time: Position exit timestamp
            ticker: Stock ticker
            price_df: Optional price DataFrame (uses cache if not provided)

        Returns:
            float: Actual return (exit_price - entry_price) / entry_price
        """
        # Use provided price_df or cached data
        if price_df is None:
            if ticker not in self.price_cache:
                raise ValueError(f"No price data loaded for {ticker}. Call load_price_data() first.")
            price_df = self.price_cache[ticker]

        try:
            # Get entry price (close of entry bar)
            if entry_time not in price_df.index:
                # Find nearest bar
                entry_time = price_df.index[price_df.index >= entry_time][0]

            entry_price = price_df.loc[entry_time, 'close']

            # Get exit price (close of exit bar)
            if exit_time not in price_df.index:
                # Find nearest bar
                exit_time = price_df.index[price_df.index >= exit_time][0]

            exit_price = price_df.loc[exit_time, 'close']

            # Calculate return
            return_pct = (exit_price - entry_price) / entry_price

            return return_pct

        except (KeyError, IndexError) as e:
            print(f"Warning: Could not calculate return for {ticker} "
                  f"from {entry_time} to {exit_time}: {e}")
            return 0.0

    def calculate_return_with_execution_price(self, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                                             ticker: str, entry_price_type: str = 'close',
                                             exit_price_type: str = 'close',
                                             price_df: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate return with specific execution prices (open, high, low, close).

        This allows more realistic execution modeling. For example:
        - Market orders: use 'open' price
        - Limit orders: use specific levels
        - Worst-case: use 'high' for buys, 'low' for sells

        Args:
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            ticker: Ticker symbol
            entry_price_type: 'open', 'high', 'low', or 'close'
            exit_price_type: 'open', 'high', 'low', or 'close'
            price_df: Optional price DataFrame

        Returns:
            float: Calculated return
        """
        if price_df is None:
            if ticker not in self.price_cache:
                raise ValueError(f"No price data for {ticker}")
            price_df = self.price_cache[ticker]

        try:
            # Get entry price
            if entry_time not in price_df.index:
                entry_time = price_df.index[price_df.index >= entry_time][0]
            entry_price = price_df.loc[entry_time, entry_price_type]

            # Get exit price
            if exit_time not in price_df.index:
                exit_time = price_df.index[price_df.index >= exit_time][0]
            exit_price = price_df.loc[exit_time, exit_price_type]

            return (exit_price - entry_price) / entry_price

        except (KeyError, IndexError) as e:
            print(f"Warning: Could not calculate return: {e}")
            return 0.0

    def calculate_returns_batch(self, trades_df: pd.DataFrame, ticker: str,
                                price_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate returns for a batch of trades.

        Args:
            trades_df: DataFrame with 'entry_time' and 'exit_time' columns
            ticker: Ticker symbol
            price_df: Optional price DataFrame

        Returns:
            Series of calculated returns
        """
        returns = []

        for idx, row in trades_df.iterrows():
            ret = self.calculate_return(
                entry_time=row['entry_time'],
                exit_time=row['exit_time'],
                ticker=ticker,
                price_df=price_df
            )
            returns.append(ret)

        return pd.Series(returns, index=trades_df.index)

    def validate_returns(self, calculated_returns: pd.Series, forward_returns: pd.Series,
                        tolerance: float = 0.01) -> Dict[str, float]:
        """
        Validate calculated returns against forward_returns (for debugging).

        This is useful during development to check if returns are being calculated correctly.
        However, in production, we should NEVER use forward_return for backtesting PnL.

        Args:
            calculated_returns: Returns calculated from actual prices
            forward_returns: forward_return column from data
            tolerance: Maximum allowed deviation (as fraction)

        Returns:
            Dictionary with validation metrics
        """
        diff = calculated_returns - forward_returns
        abs_diff = np.abs(diff)

        metrics = {
            'mean_diff': diff.mean(),
            'max_diff': diff.max(),
            'min_diff': diff.min(),
            'correlation': np.corrcoef(calculated_returns, forward_returns)[0, 1],
            'fraction_within_tolerance': (abs_diff < tolerance).mean(),
            'mean_absolute_error': abs_diff.mean()
        }

        print("\nReturn Validation Metrics:")
        print(f"  Mean difference: {metrics['mean_diff']:.6f}")
        print(f"  Max difference: {metrics['max_diff']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Within tolerance ({tolerance}): {metrics['fraction_within_tolerance']*100:.1f}%")

        if metrics['correlation'] < 0.95:
            print(f"  WARNING: Low correlation ({metrics['correlation']:.4f}) suggests calculation error!")

        return metrics


def demonstrate_return_calculation():
    """Demonstration of proper return calculation."""
    # Create sample price data
    dates = pd.date_range('2024-01-01 09:30', periods=100, freq='5min')
    np.random.seed(42)

    # Generate realistic price path
    returns = np.random.normal(0.0001, 0.01, 100)
    prices = 100 * (1 + returns).cumprod()

    price_df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
        'close': prices
    }, index=dates)

    print("Sample Price Data:")
    print(price_df.head())
    print("\n" + "="*60 + "\n")

    # Calculate return for a trade
    calculator = RealReturnCalculator()
    calculator.load_price_data(price_df, 'SPY')

    entry_time = dates[10]
    exit_time = dates[15]  # 5 bars later (25 minutes)

    calculated_return = calculator.calculate_return(entry_time, exit_time, 'SPY')

    # Manual calculation for verification
    entry_price = price_df.loc[entry_time, 'close']
    exit_price = price_df.loc[exit_time, 'close']
    manual_return = (exit_price - entry_price) / entry_price

    print(f"Trade Example:")
    print(f"  Entry time: {entry_time}")
    print(f"  Exit time: {exit_time}")
    print(f"  Entry price: ${entry_price:.2f}")
    print(f"  Exit price: ${exit_price:.2f}")
    print(f"  Calculated return: {calculated_return*100:.4f}%")
    print(f"  Manual verification: {manual_return*100:.4f}%")
    print(f"  Match: {np.isclose(calculated_return, manual_return)}")

    print("\n" + "="*60 + "\n")

    # Demonstrate why we can't use forward_return
    print("Why NOT to use forward_return column:")
    print("  - forward_return is a LABEL used for training")
    print("  - Using it in backtest creates circular logic")
    print("  - Model 'knows the answer' during backtesting")
    print("  - Results will be unrealistically optimistic")
    print("\n  ✓ Always calculate returns from actual entry/exit prices!")


if __name__ == "__main__":
    print("Real Return Calculator - Fixes Circular Backtesting Logic")
    print("="*60)
    print("\nDemonstration:\n")
    demonstrate_return_calculation()
