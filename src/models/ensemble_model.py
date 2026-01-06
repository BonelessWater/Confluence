"""
Ensemble model combining multiple trading models.

Combines predictions from Linear, XGBoost, and Attention models
using weighted averaging optimized on validation set.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.base_model import BaseTradingModel
from typing import List, Optional, Dict


class EnsembleModel(BaseTradingModel):
    """
    Ensemble of multiple trading models with optimized weights.

    Combines predictions from multiple models using weighted averaging.
    Weights are optimized on validation set to maximize correlation.

    Args:
        name: Model name
        models: List of fitted BaseTradingModel instances
        weights: Optional list of weights (optimized if None)
        method: Ensemble method ('weighted_average', 'median', 'mean')
    """

    def __init__(self, name: str = "Ensemble",
                 models: Optional[List[BaseTradingModel]] = None,
                 weights: Optional[List[float]] = None,
                 method: str = 'weighted_average'):
        super().__init__(name)
        self.models = models or []
        self.weights = weights
        self.method = method
        self.optimized_weights = None

        if self.weights is not None:
            # Validate weights
            if len(self.weights) != len(self.models):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of models ({len(self.models)})"
                )
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1, got {sum(self.weights)}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs):
        """
        Optimize ensemble weights on validation set.

        Note: Individual models should already be fitted.
        This method only optimizes the ensemble weights.

        Args:
            X_train: Training features (not used, but required by interface)
            y_train: Training labels (not used, but required by interface)
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional arguments

        Returns:
            Dictionary with ensemble metrics
        """
        print(f"\n{'='*80}")
        print(f"OPTIMIZING {self.name} WEIGHTS")
        print(f"{'='*80}")
        print(f"Number of models: {len(self.models)}")
        print(f"Method: {self.method}")

        # Check all models are fitted
        for i, model in enumerate(self.models):
            if not model.is_fitted:
                raise ValueError(
                    f"Model {i} ({model.name}) must be fitted before ensemble"
                )

        if self.method == 'weighted_average':
            if self.weights is None:
                # Optimize weights
                print("\nOptimizing weights on validation set...")
                self.weights = self._optimize_weights(X_val, y_val)
                self.optimized_weights = self.weights.copy()

                print("\nOptimized Weights:")
                for model, weight in zip(self.models, self.weights):
                    print(f"  {model.name}: {weight:.4f}")
            else:
                print("\nUsing pre-specified weights:")
                for model, weight in zip(self.models, self.weights):
                    print(f"  {model.name}: {weight:.4f}")
        elif self.method in ['median', 'mean']:
            print(f"\nUsing {self.method} (no weight optimization needed)")
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True

        # Evaluate ensemble
        print("\nEvaluating ensemble...")
        val_metrics = self.evaluate(X_val, y_val)

        print(f"\nEnsemble Performance:")
        print(f"  Correlation: {val_metrics['correlation']:.4f}")
        print(f"  MSE: {val_metrics['mse']:.6f}")
        print(f"  MAE: {val_metrics['mae']:.6f}")
        print(f"  Directional Accuracy: {val_metrics['directional_accuracy']*100:.2f}%")

        # Compare to individual models
        print("\nComparison to Individual Models:")
        for model in self.models:
            model_metrics = model.evaluate(X_val, y_val)
            print(f"  {model.name}: Corr={model_metrics['correlation']:.4f}, "
                  f"Dir Acc={model_metrics['directional_accuracy']*100:.1f}%")

        improvement = val_metrics['correlation'] - np.mean(
            [m.evaluate(X_val, y_val)['correlation'] for m in self.models]
        )

        if improvement > 0:
            print(f"\n✓ Ensemble improves over average by {improvement:.4f}")
        else:
            print(f"\n⚠️ Ensemble correlation {improvement:.4f} vs average")

        print(f"{'='*80}")

        return {
            'val_metrics': val_metrics,
            'weights': self.weights,
            'method': self.method
        }

    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> List[float]:
        """
        Optimize weights to maximize validation correlation.

        Uses scipy.optimize to find optimal weights.

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Optimized weights
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X_val)
            predictions.append(pred)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)

        # Try scipy optimization if available
        try:
            from scipy.optimize import minimize

            def objective(weights):
                """Negative correlation (minimize to maximize correlation)."""
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                corr = np.corrcoef(y_val, ensemble_pred)[0, 1]
                return -corr  # Negative because we minimize

            # Constraints: weights sum to 1, all non-negative
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(len(self.models))]

            # Initial guess: equal weights
            w0 = np.array([1.0 / len(self.models)] * len(self.models))

            # Optimize
            result = minimize(
                objective, w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )

            if result.success:
                optimized_weights = result.x.tolist()
                print(f"Optimization converged (correlation: {-result.fun:.4f})")
                return optimized_weights
            else:
                print(f"Optimization failed, using equal weights")
                return [1.0 / len(self.models)] * len(self.models)

        except ImportError:
            print("scipy not available, using equal weights")
            return [1.0 / len(self.models)] * len(self.models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features to predict on

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)

        # Combine based on method
        if self.method == 'weighted_average':
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        elif self.method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
        elif self.method == 'mean':
            ensemble_pred = np.mean(predictions, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return ensemble_pred

    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions and their contributions.

        Args:
            X: Features

        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")

        contributions = {}
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            contributions[model.name] = {
                'prediction': pred,
                'weight': weight,
                'weighted_prediction': pred * weight
            }

        return contributions

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        n_models = len(self.models)
        model_names = [m.name for m in self.models]

        return (f"EnsembleModel(name='{self.name}', n_models={n_models}, "
                f"method='{self.method}', status={status}, "
                f"models={model_names})")


def demonstrate_ensemble():
    """Demonstration of ensemble model."""
    print("Ensemble Model Demonstration")
    print("="*80)

    # This is a conceptual demonstration
    # In practice, you'd use real trained models

    from src.models.linear_model import LinearModel
    from src.models.xgboost_model import XGBoostModel

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    # Complex relationship
    y = (
        0.3 * X[:, 0] +
        0.2 * X[:, 1] ** 2 +
        0.1 * X[:, 2] * X[:, 3] +
        np.random.randn(n_samples) * 0.1
    )

    # Split
    split_idx = int(n_samples * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train individual models
    print("\nTraining individual models...")
    linear = LinearModel(name="Linear", n_features=20)
    xgb_model = XGBoostModel(name="XGBoost", max_depth=5, n_estimators=50)

    linear.fit(X_train, y_train, X_val, y_val)
    xgb_model.fit(X_train, y_train, X_val, y_val)

    # Create ensemble
    print("\n" + "="*80)
    ensemble = EnsembleModel(
        name="Ensemble_Linear_XGB",
        models=[linear, xgb_model],
        method='weighted_average'
    )

    # Fit ensemble (optimize weights)
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Make predictions
    ensemble_pred = ensemble.predict(X_val)

    print(f"\nSample Predictions:")
    print(f"  True: {y_val[:5]}")
    print(f"  Ensemble: {ensemble_pred[:5]}")

    print("\n" + "="*80)
    print("Ensemble demonstration complete!")


if __name__ == "__main__":
    demonstrate_ensemble()
