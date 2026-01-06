"""
Transaction cost calculator for realistic backtesting.

This module adds commission and slippage costs to trades, providing
realistic performance estimates.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from config.settings import COMMISSION_BPS, SLIPPAGE_BPS, TOTAL_COST_BPS


class TransactionCostCalculator:
    """
    Calculate transaction costs for trades.

    Transaction costs include:
    1. Commission: Broker fees per trade (default 5 bps)
    2. Slippage: Market impact and execution slippage (default 3 bps)

    Total: 8 bps per trade (one-way), 16 bps round-trip
    """

    def __init__(self, commission_bps: float = COMMISSION_BPS,
                 slippage_bps: float = SLIPPAGE_BPS):
        """
        Initialize transaction cost calculator.

        Args:
            commission_bps: Commission in basis points (default 5 bps)
            slippage_bps: Slippage in basis points (default 3 bps)
        """
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.total_bps = commission_bps + slippage_bps

        print(f"Transaction Costs initialized:")
        print(f"  Commission: {commission_bps} bps")
        print(f"  Slippage: {slippage_bps} bps")
        print(f"  Total (one-way): {self.total_bps} bps")
        print(f"  Round-trip: {self.total_bps * 2} bps")

    def calculate_one_way_cost(self, position_size: float) -> float:
        """
        Calculate one-way transaction cost (entry OR exit).

        Args:
            position_size: Dollar amount of position

        Returns:
            float: Transaction cost in dollars
        """
        return position_size * (self.total_bps / 10000.0)

    def calculate_round_trip_cost(self, position_size: float) -> float:
        """
        Calculate round-trip transaction cost (entry + exit).

        Args:
            position_size: Dollar amount of position

        Returns:
            float: Total round-trip cost in dollars
        """
        return position_size * (self.total_bps / 10000.0) * 2

    def apply_costs_to_return(self, gross_return: float, position_weight: float = 1.0) -> float:
        """
        Apply round-trip transaction costs to gross return.

        Args:
            gross_return: Return before costs (as decimal, e.g., 0.02 for 2%)
            position_weight: Portfolio weight (0-1, default 1.0 for full position)

        Returns:
            float: Net return after costs

        Example:
            >>> calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)
            >>> gross = 0.02  # 2% gross return
            >>> net = calc.apply_costs_to_return(gross, position_weight=1.0)
            >>> # Net = 2.0% - 0.16% = 1.84%
        """
        # Round-trip cost: entry + exit
        round_trip_cost_pct = (self.total_bps / 10000.0) * 2

        # Apply costs (proportional to position size)
        net_return = gross_return - (round_trip_cost_pct * position_weight)

        return net_return

    def calculate_cost_breakdown(self, gross_return: float, position_size: float) -> Dict[str, float]:
        """
        Calculate detailed cost breakdown for a trade.

        Args:
            gross_return: Gross return (as decimal)
            position_size: Dollar size of position

        Returns:
            Dictionary with detailed cost breakdown
        """
        # Calculate costs
        commission_entry = position_size * (self.commission_bps / 10000.0)
        slippage_entry = position_size * (self.slippage_bps / 10000.0)
        entry_cost = commission_entry + slippage_entry

        commission_exit = position_size * (self.commission_bps / 10000.0)
        slippage_exit = position_size * (self.slippage_bps / 10000.0)
        exit_cost = commission_exit + slippage_exit

        total_cost = entry_cost + exit_cost

        # Calculate P&L
        gross_pnl = position_size * gross_return
        net_pnl = gross_pnl - total_cost

        return {
            'commission_entry': commission_entry,
            'slippage_entry': slippage_entry,
            'entry_cost': entry_cost,
            'commission_exit': commission_exit,
            'slippage_exit': slippage_exit,
            'exit_cost': exit_cost,
            'total_cost': total_cost,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'cost_as_pct_of_gross': (total_cost / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        }

    def get_breakeven_return(self) -> float:
        """
        Calculate minimum return needed to break even after costs.

        Returns:
            float: Breakeven return (as decimal)
        """
        breakeven = (self.total_bps / 10000.0) * 2
        return breakeven

    def estimate_annual_cost_drag(self, avg_holding_period_days: float,
                                  trades_per_year: Optional[int] = None) -> float:
        """
        Estimate annual cost drag from transaction costs.

        Args:
            avg_holding_period_days: Average holding period in days
            trades_per_year: Number of trades per year (calculated if not provided)

        Returns:
            float: Estimated annual cost drag (as decimal)
        """
        if trades_per_year is None:
            # Estimate trades per year from holding period
            trades_per_year = 252 / avg_holding_period_days  # 252 trading days

        # Annual cost = trades/year * round-trip cost
        annual_cost = trades_per_year * (self.total_bps / 10000.0) * 2

        return annual_cost


class TieredCostCalculator(TransactionCostCalculator):
    """
    Transaction cost calculator with tiered pricing based on trade size or volume.

    Larger trades or higher volume traders typically get better rates.
    """

    def __init__(self, cost_tiers: Dict[float, tuple] = None):
        """
        Initialize tiered cost calculator.

        Args:
            cost_tiers: Dictionary mapping volume thresholds to (commission, slippage) tuples
                       e.g., {0: (5, 3), 10000: (3, 2), 100000: (1, 1)}
        """
        if cost_tiers is None:
            # Default tiers
            cost_tiers = {
                0: (5.0, 3.0),        # < $10k: 5 bps commission, 3 bps slippage
                10000: (3.0, 2.0),    # $10k-$100k: 3 bps commission, 2 bps slippage
                100000: (1.0, 1.0)    # > $100k: 1 bps commission, 1 bps slippage
            }

        self.cost_tiers = cost_tiers
        # Initialize with base tier
        base_commission, base_slippage = cost_tiers[0]
        super().__init__(base_commission, base_slippage)

    def get_costs_for_size(self, position_size: float) -> tuple:
        """
        Get commission and slippage rates for given position size.

        Args:
            position_size: Dollar size of position

        Returns:
            (commission_bps, slippage_bps) tuple
        """
        # Find appropriate tier
        applicable_tier = 0
        for threshold in sorted(self.cost_tiers.keys(), reverse=True):
            if position_size >= threshold:
                applicable_tier = threshold
                break

        return self.cost_tiers[applicable_tier]

    def calculate_one_way_cost(self, position_size: float) -> float:
        """Calculate one-way cost with tiered pricing."""
        commission_bps, slippage_bps = self.get_costs_for_size(position_size)
        total_bps = commission_bps + slippage_bps
        return position_size * (total_bps / 10000.0)


def demonstrate_transaction_costs():
    """Demonstration of transaction cost calculations."""
    print("Transaction Cost Examples")
    print("="*60 + "\n")

    # Initialize calculator
    calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)

    print("\nExample 1: $10,000 position with 2% gross return")
    print("-"*60)
    position_size = 10000
    gross_return = 0.02  # 2%

    breakdown = calc.calculate_cost_breakdown(gross_return, position_size)
    print(f"Position size: ${position_size:,.0f}")
    print(f"Gross return: {gross_return*100:.2f}%")
    print(f"Gross P&L: ${breakdown['gross_pnl']:.2f}")
    print(f"\nCosts:")
    print(f"  Entry commission: ${breakdown['commission_entry']:.2f}")
    print(f"  Entry slippage: ${breakdown['slippage_entry']:.2f}")
    print(f"  Exit commission: ${breakdown['commission_exit']:.2f}")
    print(f"  Exit slippage: ${breakdown['slippage_exit']:.2f}")
    print(f"  Total cost: ${breakdown['total_cost']:.2f} ({breakdown['cost_as_pct_of_gross']:.1f}% of gross)")
    print(f"\nNet P&L: ${breakdown['net_pnl']:.2f}")
    net_return = calc.apply_costs_to_return(gross_return)
    print(f"Net return: {net_return*100:.2f}%")

    print("\n" + "="*60 + "\n")

    print("Example 2: Breakeven analysis")
    print("-"*60)
    breakeven = calc.get_breakeven_return()
    print(f"Breakeven return (after costs): {breakeven*100:.4f}%")
    print(f"  (Minimum return needed to not lose money after costs)")

    print("\n" + "="*60 + "\n")

    print("Example 3: Annual cost drag estimate")
    print("-"*60)
    avg_holding_days = 0.0035  # ~5 minutes for 5-min holding period
    annual_drag = calc.estimate_annual_cost_drag(avg_holding_days)
    print(f"Average holding period: {avg_holding_days*24*60:.1f} minutes")
    print(f"Estimated trades/year: {252/avg_holding_days:,.0f}")
    print(f"Estimated annual cost drag: {annual_drag*100:.1f}%")
    print(f"  (This is why frequent trading is expensive!)")

    print("\n" + "="*60 + "\n")

    print("Example 4: Impact on different return scenarios")
    print("-"*60)
    scenarios = [
        ("Small loss", -0.005),
        ("Breakeven", 0.0),
        ("Small win", 0.005),
        ("Medium win", 0.01),
        ("Large win", 0.02)
    ]

    print(f"{'Scenario':<15} {'Gross':<10} {'Net':<10} {'Cost Impact'}")
    print("-"*60)
    for name, gross in scenarios:
        net = calc.apply_costs_to_return(gross)
        impact = gross - net
        print(f"{name:<15} {gross*100:>6.2f}%   {net*100:>6.2f}%   {impact*100:>6.2f}%")


if __name__ == "__main__":
    demonstrate_transaction_costs()
