"""
XGBoost gradient boosting baseline model.

XGBoost is a powerful gradient boosting framework that often performs
well on tabular data. This serves as a strong non-linear baseline.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.base_model import BaseTradingModel
import xgboost as xgb
from config.settings import (
    XGBOOST_MAX_DEPTH, XGBOOST_N_ESTIMATORS,
    XGBOOST_LEARNING_RATE, XGBOOST_EARLY_STOPPING_ROUNDS
)


class XGBoostModel(BaseTradingModel):
    """
    XGBoost regression model for trading predictions.

    Gradient boosting is effective for capturing non-linear relationships
    and feature interactions without manual feature engineering.

    Args:
        name: Model name
        max_depth: Maximum tree depth (default from config)
        n_estimators: Number of boosting rounds (default from config)
        learning_rate: Learning rate / eta (default from config)
        early_stopping_rounds: Early stopping patience (default from config)
    """

    def __init__(self, name: str = "XGBoost",
                 max_depth: int = XGBOOST_MAX_DEPTH,
                 n_estimators: int = XGBOOST_N_ESTIMATORS,
                 learning_rate: float = XGBOOST_LEARNING_RATE,
                 early_stopping_rounds: int = XGBOOST_EARLY_STOPPING_ROUNDS):
        super().__init__(name)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.best_iteration = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs):
        """
        Train XGBoost model with early stopping.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional XGBoost parameters

        Returns:
            Dictionary with training metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING {self.name}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Max depth: {self.max_depth}")
        print(f"  N estimators: {self.n_estimators}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Early stopping rounds: {self.early_stopping_rounds}")
        print(f"  Input features: {X_train.shape[1]}")

        # Initialize XGBoost model
        print(f"\nInitializing XGBoost regressor...")
        self.model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            objective='reg:squarederror',
            tree_method='hist',  # Fast histogram-based algorithm
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            **kwargs
        )

        # Train with early stopping
        print(f"Training with early stopping...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        self.is_fitted = True

        # Evaluate on both sets
        print(f"\nEvaluating model...")
        train_metrics = self.evaluate(X_train, y_train)
        val_metrics = self.evaluate(X_val, y_val)

        print(f"\nTraining Metrics:")
        print(f"  Correlation: {train_metrics['correlation']:.4f}")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  MAE: {train_metrics['mae']:.6f}")
        print(f"  Directional Accuracy: {train_metrics['directional_accuracy']*100:.2f}%")

        print(f"\nValidation Metrics:")
        print(f"  Correlation: {val_metrics['correlation']:.4f}")
        print(f"  MSE: {val_metrics['mse']:.6f}")
        print(f"  MAE: {val_metrics['mae']:.6f}")
        print(f"  Directional Accuracy: {val_metrics['directional_accuracy']*100:.2f}%")

        print(f"\nBoosting Info:")
        print(f"  Best iteration: {self.best_iteration}")
        print(f"  Total trees: {self.model.n_estimators}")

        # Check for overfitting
        corr_diff = train_metrics['correlation'] - val_metrics['correlation']
        if corr_diff > 0.15:
            print(f"\n⚠️ WARNING: Possible overfitting detected!")
            print(f"  Training correlation ({train_metrics['correlation']:.4f}) >> "
                  f"Validation correlation ({val_metrics['correlation']:.4f})")
            print(f"  Difference: {corr_diff:.4f}")
            print(f"  Consider: Lower max_depth, increase regularization, or more early stopping")
        elif corr_diff > 0.1:
            print(f"\n⚠️ CAUTION: Moderate overfitting detected (difference: {corr_diff:.4f})")
        else:
            print(f"\n✓ Good generalization (correlation difference: {corr_diff:.4f})")

        print(f"{'='*80}")

        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_iteration': self.best_iteration
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        predictions = self.model.predict(X)
        return predictions

    def get_feature_importance(self, feature_names: list = None,
                               importance_type: str = 'gain') -> np.ndarray:
        """
        Get feature importance.

        Args:
            feature_names: Optional list of feature names
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Array or dictionary of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get importance scores
        importance = self.model.feature_importances_

        if feature_names is not None:
            # Create dictionary mapping
            importance_dict = dict(zip(feature_names, importance))
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(),
                                      key=lambda x: x[1],
                                      reverse=True)
            return sorted_importance

        return importance

    def plot_feature_importance(self, feature_names: list = None, top_k: int = 20):
        """
        Plot top K most important features.

        Args:
            feature_names: Optional list of feature names
            top_k: Number of top features to plot
        """
        try:
            import matplotlib.pyplot as plt

            importance = self.get_feature_importance(feature_names)

            if feature_names is not None:
                # Sorted list of tuples
                top_features = importance[:top_k]
                names = [f[0] for f in top_features]
                scores = [f[1] for f in top_features]
            else:
                # Array
                sorted_idx = np.argsort(importance)[::-1][:top_k]
                names = [f"Feature {i}" for i in sorted_idx]
                scores = importance[sorted_idx]

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(names)), scores)
            plt.yticks(range(len(names)), names)
            plt.xlabel('Importance Score')
            plt.title(f'Top {top_k} Most Important Features - {self.name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib not available for plotting")

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"XGBoostModel(name='{self.name}', max_depth={self.max_depth}, "
                f"n_estimators={self.n_estimators}, lr={self.learning_rate}, "
                f"status={status})")


def demonstrate_xgboost_model():
    """Demonstration of XGBoost model."""
    print("XGBoost Model Baseline")
    print("="*80)

    # Generate synthetic data with non-linear relationships
    np.random.seed(42)
    n_samples = 1000
    n_features = 100

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create target with non-linear relationships
    # Use polynomial and interaction terms
    y = (
        0.5 * X[:, 0] ** 2 +  # Quadratic
        0.3 * X[:, 1] * X[:, 2] +  # Interaction
        0.2 * np.sin(X[:, 3] * 2) +  # Non-linear
        0.1 * X[:, 4] +  # Linear
        np.random.randn(n_samples) * 0.1  # Noise
    )

    # Split data
    split_idx = int(n_samples * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train model
    model = XGBoostModel(
        name="XGBoost_Baseline",
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        early_stopping_rounds=10
    )

    results = model.fit(X_train, y_train, X_val, y_val)

    # Make predictions
    predictions = model.predict(X_val)

    print(f"\nPrediction Examples:")
    print(f"  True: {y_val[:5]}")
    print(f"  Pred: {predictions[:5]}")

    # Feature importance
    print(f"\nTop 10 Most Important Features:")
    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)[::-1][:10]
    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1}. Feature {idx}: {importance[idx]:.6f}")

    # Verify it captured the key relationships
    print(f"\nKey Features (should be in top 5):")
    print(f"  Feature 0 (quadratic): Rank {np.where(sorted_idx == 0)[0][0] + 1 if 0 in sorted_idx else 'Not in top 10'}")
    print(f"  Feature 1 (interaction): Rank {np.where(sorted_idx == 1)[0][0] + 1 if 1 in sorted_idx else 'Not in top 10'}")
    print(f"  Feature 2 (interaction): Rank {np.where(sorted_idx == 2)[0][0] + 1 if 2 in sorted_idx else 'Not in top 10'}")
    print(f"  Feature 3 (sine): Rank {np.where(sorted_idx == 3)[0][0] + 1 if 3 in sorted_idx else 'Not in top 10'}")
    print(f"  Feature 4 (linear): Rank {np.where(sorted_idx == 4)[0][0] + 1 if 4 in sorted_idx else 'Not in top 10'}")

    print("\n" + "="*80)
    print("XGBoost model demonstration complete!")


if __name__ == "__main__":
    demonstrate_xgboost_model()
