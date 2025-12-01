"""
Metrics for evaluating predictive power of L1 consolidation methods.

This module provides functions to calculate accuracy and efficiency metrics
for different price estimation methods.
"""

import numpy as np
import time
from typing import Dict, Tuple


class PredictionMetrics:
    """Calculate and store prediction performance metrics."""

    @staticmethod
    def calculate_accuracy_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate accuracy metrics for predictions.

        Args:
            actual: Actual prices (1D array)
            predicted: Predicted prices (1D array)

        Returns:
            Dict with MAE, RMSE, MAPE, and max error
        """
        errors = np.abs(actual - predicted)
        mae = errors.mean()
        rmse = np.sqrt(np.mean(errors**2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        max_error = errors.max()

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'max_error': max_error
        }

    @staticmethod
    def calculate_quality_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate variance explanation and correlation metrics.

        Args:
            actual: Actual prices (1D array)
            predicted: Predicted prices (1D array)

        Returns:
            Dict with R-squared and correlation coefficient
        """
        actual_mean = actual.mean()
        ss_tot = np.sum((actual - actual_mean)**2)
        ss_res = np.sum((actual - predicted)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        correlation = np.corrcoef(actual, predicted)[0, 1]

        return {
            'r_squared': r_squared,
            'correlation': correlation
        }

    @staticmethod
    def measure_computation_time(func, *args, num_iterations: int = 100, **kwargs) -> float:
        """
        Measure average computation time for a function.

        Args:
            func: Function to time
            *args: Positional arguments to func
            num_iterations: Number of times to call the function
            **kwargs: Keyword arguments to func

        Returns:
            Average time per call in seconds
        """
        start = time.perf_counter()
        for _ in range(num_iterations):
            func(*args, **kwargs)
        return (time.perf_counter() - start) / num_iterations

    @staticmethod
    def rank_methods(results: Dict[str, Dict]) -> list:
        """
        Rank methods based on weighted combination of metrics.

        Weights: 50% prediction error (MAE), 35% quality (R²), 15% efficiency (time)

        Args:
            results: Dict with method names as keys, each containing 'mae', 'r_squared', 'time_us'

        Returns:
            List of (method_name, overall_score) tuples sorted by score descending
        """
        methods = list(results.keys())

        # Extract metrics
        mae_values = [results[m]['mae'] for m in methods]
        r2_values = [results[m]['r_squared'] for m in methods]
        time_values = [results[m]['time_us'] for m in methods]

        # Normalize to 0-10 scale
        min_mae, max_mae = min(mae_values), max(mae_values)
        min_r2, max_r2 = min(r2_values), max(r2_values)
        min_time, max_time = min(time_values), max(time_values)

        scores = {}
        for method in methods:
            # Lower MAE = higher score
            mae_score = 10.0 * (max_mae - results[method]['mae']) / (max_mae - min_mae) if max_mae != min_mae else 5.0
            # Higher R² = higher score
            r2_score = 10.0 * (results[method]['r_squared'] - min_r2) / (max_r2 - min_r2) if max_r2 != min_r2 else 5.0
            # Lower time = higher score
            time_score = 10.0 * (max_time - results[method]['time_us']) / (max_time - min_time) if max_time != min_time else 5.0

            # Weighted combination
            overall = (mae_score * 0.50) + (r2_score * 0.35) + (time_score * 0.15)
            scores[method] = overall

        # Sort by score descending
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranking


def evaluate_method(actual: np.ndarray, predicted: np.ndarray,
                   computation_time_us: float) -> Dict[str, float]:
    """
    Comprehensive evaluation of a prediction method.

    Args:
        actual: Actual prices
        predicted: Predicted prices
        computation_time_us: Time in microseconds

    Returns:
        Dict with all metrics
    """
    metrics = {}
    metrics.update(PredictionMetrics.calculate_accuracy_metrics(actual, predicted))
    metrics.update(PredictionMetrics.calculate_quality_metrics(actual, predicted))
    metrics['time_us'] = computation_time_us
    metrics['predictions'] = len(actual)

    return metrics
