"""
Master test runner for all tests.

Runs all test suites and provides summary.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_features import run_feature_tests
from test_backtest import run_backtest_tests
from test_models import run_model_tests


def run_all_tests():
    """Run all test suites."""
    print("="*80)
    print("RUNNING ALL TESTS")
    print("="*80)

    results = {}

    # Run feature tests
    print("\n" + "="*80)
    print("1. FEATURE TESTS (Look-ahead Bias Detection)")
    print("="*80)
    results['features'] = run_feature_tests()

    # Run backtest tests
    print("\n" + "="*80)
    print("2. BACKTEST TESTS (Return Calculation & Costs)")
    print("="*80)
    results['backtest'] = run_backtest_tests()

    # Run model tests
    print("\n" + "="*80)
    print("3. MODEL TESTS (Interface & Performance)")
    print("="*80)
    results['models'] = run_model_tests()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for test_suite, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_suite.upper()}: {status}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nCritical verifications:")
        print("  ✓ No look-ahead bias in features")
        print("  ✓ Correct return calculation (no circular logic)")
        print("  ✓ Transaction costs applied correctly")
        print("  ✓ Models implement required interface")
        print("\nSystem is ready for backtesting!")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("\nPlease review failed tests before running backtests.")

    print("="*80)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
