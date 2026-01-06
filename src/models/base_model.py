"""
Base model class for modular trading model architecture.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
from typing import Tuple, Dict, Any

class BaseTradingModel(ABC):
    """
    Abstract base class for trading models.
    All models must implement this interface.
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_columns = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions array
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)

        # Calculate metrics
        correlation = np.corrcoef(y, predictions)[0, 1]
        mse = np.mean((y - predictions) ** 2)
        mae = np.mean(np.abs(y - predictions))

        # Directional accuracy
        y_direction = np.sign(y)
        pred_direction = np.sign(predictions)
        directional_accuracy = np.mean(y_direction == pred_direction)

        return {
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }

    def benchmark_inference_speed(self, X: np.ndarray, num_runs: int = 1000) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            X: Sample data
            num_runs: Number of inference runs

        Returns:
            Dictionary with timing statistics
        """
        times = []

        # Warm-up
        for _ in range(10):
            _ = self.predict(X[:1])

        # Benchmark
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(X[:1])
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)

        return {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'max_ms': np.max(times)
        }

    def save(self, path: str):
        """Save model to disk."""
        import joblib
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'name': self.name
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.name = data['name']
        self.is_fitted = True
        print(f"Model loaded from {path}")

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
