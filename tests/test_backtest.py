"""
Tests for backtesting components.

CRITICAL: These tests verify that returns are calculated correctly
and transaction costs are applied properly.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.backtesting.return_calculator import RealReturnCalculator
from src.backtesting.transaction_costs import TransactionCostCalculator


class TestReturnCalculator(unittest.TestCase):
    """Test return calculation from prices."""

    def setUp(self):
        """Create sample price data."""
        dates = pd.date_range('2024-01-01 09:30', periods=100, freq='5min')
        self.price_df = pd.DataFrame({
            'open': np.arange(100, 200) + np.random.randn(100) * 0.1,
            'high': np.arange(100, 200) + 1 + np.random.randn(100) * 0.1,
            'low': np.arange(100, 200) - 1 + np.random.randn(100) * 0.1,
            'close': np.arange(100, 200)
        }, index=dates)

        self.calculator = RealReturnCalculator()
        self.calculator.load_price_data(self.price_df, 'TEST')

    def test_return_calculation_accuracy(self):
        """CRITICAL: Verify return calculation is correct."""
        entry_time = self.price_df.index[10]
        exit_time = self.price_df.index[15]

        calculated_return = self.calculator.calculate_return(
            entry_time, exit_time, 'TEST'
        )

        # Manual calculation
        entry_price = self.price_df.loc[entry_time, 'close']
        exit_price = self.price_df.loc[exit_time, 'close']
        expected_return = (exit_price - entry_price) / entry_price

        self.assertAlmostEqual(
            calculated_return,
            expected_return,
            places=10,
            msg=f"Return calculation mismatch: "
                f"expected {expected_return}, got {calculated_return}"
        )

    def test_no_circular_logic(self):
        """CRITICAL: Ensure we're not using forward_return column."""
        # This test verifies the calculator uses actual prices,
        # not training labels

        entry_time = self.price_df.index[20]
        exit_time = self.price_df.index[25]

        # Calculate return
        calculated = self.calculator.calculate_return(
            entry_time, exit_time, 'TEST'
        )

        # Verify it's based on prices
        entry_price = self.price_df.loc[entry_time, 'close']
        exit_price = self.price_df.loc[exit_time, 'close']
        expected = (exit_price - entry_price) / entry_price

        self.assertAlmostEqual(calculated, expected, places=10)

        # Verify it's NOT zero (which would suggest it's using placeholders)
        self.assertNotEqual(calculated, 0.0)

    def test_zero_return_when_prices_equal(self):
        """Test return is zero when entry and exit prices are equal."""
        # Create flat price series
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        flat_prices = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10
        }, index=dates)

        calc = RealReturnCalculator()
        calc.load_price_data(flat_prices, 'FLAT')

        ret = calc.calculate_return(dates[0], dates[5], 'FLAT')

        self.assertAlmostEqual(ret, 0.0, places=10)

    def test_positive_return(self):
        """Test positive return calculation."""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        increasing_prices = pd.DataFrame({
            'open': np.arange(100, 110),
            'high': np.arange(101, 111),
            'low': np.arange(99, 109),
            'close': np.arange(100, 110)
        }, index=dates)

        calc = RealReturnCalculator()
        calc.load_price_data(increasing_prices, 'UP')

        ret = calc.calculate_return(dates[0], dates[5], 'UP')

        # Entry: 100, Exit: 105, Return: 5%
        expected = (105 - 100) / 100
        self.assertAlmostEqual(ret, expected, places=10)
        self.assertGreater(ret, 0)

    def test_negative_return(self):
        """Test negative return calculation."""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        decreasing_prices = pd.DataFrame({
            'open': np.arange(110, 100, -1),
            'high': np.arange(111, 101, -1),
            'low': np.arange(109, 99, -1),
            'close': np.arange(110, 100, -1)
        }, index=dates)

        calc = RealReturnCalculator()
        calc.load_price_data(decreasing_prices, 'DOWN')

        ret = calc.calculate_return(dates[0], dates[5], 'DOWN')

        # Entry: 110, Exit: 105, Return: -4.55%
        expected = (105 - 110) / 110
        self.assertAlmostEqual(ret, expected, places=10)
        self.assertLess(ret, 0)


class TestTransactionCosts(unittest.TestCase):
    """Test transaction cost calculations."""

    def test_cost_calculation(self):
        """Verify transaction cost calculation is correct."""
        calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)

        position_size = 10000.0  # $10k position
        cost = calc.calculate_one_way_cost(position_size)

        # Expected: 8 bps = 0.0008 * 10000 = $8
        expected_cost = 10000.0 * (8.0 / 10000.0)

        self.assertAlmostEqual(cost, expected_cost, places=2)

    def test_round_trip_cost(self):
        """Verify round-trip cost is 2x one-way."""
        calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)

        position_size = 10000.0
        one_way = calc.calculate_one_way_cost(position_size)
        round_trip = calc.calculate_round_trip_cost(position_size)

        self.assertAlmostEqual(round_trip, one_way * 2, places=2)

    def test_apply_costs_to_return(self):
        """Test applying costs reduces return."""
        calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)

        gross_return = 0.02  # 2% gross return
        net_return = calc.apply_costs_to_return(gross_return, position_weight=1.0)

        # Round-trip cost: 8 bps * 2 = 16 bps = 0.0016
        expected_net = 0.02 - 0.0016  # 1.84%

        self.assertAlmostEqual(net_return, expected_net, places=4)
        self.assertLess(net_return, gross_return, "Net return should be less than gross")

    def test_costs_can_make_profit_into_loss(self):
        """Test that costs can turn small profit into loss."""
        calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)

        # Small profit: 0.1%
        gross_return = 0.001
        net_return = calc.apply_costs_to_return(gross_return)

        # Round-trip cost: 16 bps = 0.16% > 0.1% profit
        # So net should be negative
        self.assertLess(net_return, 0, "Small profit should become loss after costs")

    def test_breakeven_return(self):
        """Test breakeven return calculation."""
        calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)

        breakeven = calc.get_breakeven_return()

        # Breakeven should be round-trip cost: 16 bps
        expected = (8.0 / 10000.0) * 2

        self.assertAlmostEqual(breakeven, expected, places=6)

    def test_cost_breakdown(self):
        """Test detailed cost breakdown."""
        calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)

        gross_return = 0.05  # 5% return
        position_size = 10000

        breakdown = calc.calculate_cost_breakdown(gross_return, position_size)

        # Verify components
        self.assertAlmostEqual(
            breakdown['entry_cost'] + breakdown['exit_cost'],
            breakdown['total_cost'],
            places=2
        )

        self.assertAlmostEqual(
            breakdown['gross_pnl'] - breakdown['total_cost'],
            breakdown['net_pnl'],
            places=2
        )

        # Verify costs are reasonable (should be around $16 for round-trip)
        self.assertAlmostEqual(breakdown['total_cost'], 16.0, places=1)


class TestBacktestIntegration(unittest.TestCase):
    """Integration tests for backtest components."""

    def test_return_and_cost_integration(self):
        """Test that returns and costs work together correctly."""
        # Create price data
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        price_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        }, index=dates)

        # Calculate return
        ret_calc = RealReturnCalculator()
        ret_calc.load_price_data(price_df, 'TEST')

        gross_return = ret_calc.calculate_return(dates[0], dates[5], 'TEST')

        # Apply costs
        cost_calc = TransactionCostCalculator(commission_bps=5.0, slippage_bps=3.0)
        net_return = cost_calc.apply_costs_to_return(gross_return)

        # Verify net < gross
        self.assertLess(net_return, gross_return)

        # Verify gross return is correct
        expected_gross = (105 - 100) / 100  # 5%
        self.assertAlmostEqual(gross_return, expected_gross, places=10)

        # Verify net return is correct (5% - 0.16%)
        expected_net = expected_gross - 0.0016
        self.assertAlmostEqual(net_return, expected_net, places=4)


def run_backtest_tests():
    """Run all backtest tests."""
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*80)
    print("BACKTEST COMPONENT TESTS")
    print("="*80)
    print("\nCRITICAL: These tests verify correct return calculation and costs\n")

    success = run_backtest_tests()

    print("\n" + "="*80)
    if success:
        print("✓ ALL BACKTEST TESTS PASSED - Returns and costs working correctly")
    else:
        print("❌ SOME TESTS FAILED - Review backtest components")
    print("="*80)

    sys.exit(0 if success else 1)
