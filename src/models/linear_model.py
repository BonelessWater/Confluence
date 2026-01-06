"""
Linear regression baseline model with feature selection.

This simple baseline helps assess whether complex models (like attention)
provide meaningful improvement over linear methods.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.base_model import BaseTradingModel
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from config.settings import LINEAR_N_FEATURES, LINEAR_ALPHA


class LinearModel(BaseTradingModel):
    """
    Linear regression baseline with feature selection.

    Uses Ridge regression (L2 regularization) on top K features selected
    by univariate F-test. This provides a simple, interpretable baseline.

    Args:
        name: Model name
        n_features: Number of top features to select (default from config)
        alpha: Ridge regularization strength (default from config)
    """

    def __init__(self, name: str = "LinearRegression",
                 n_features: int = LINEAR_N_FEATURES,
                 alpha: float = LINEAR_ALPHA):
        super().__init__(name)
        self.n_features = n_features
        self.alpha = alpha
        self.feature_selector = None
        self.model = None
        self.selected_feature_indices = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs):
        """
        Train linear model with feature selection.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with training metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING {self.name}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Number of features to select: {self.n_features}")
        print(f"  Ridge alpha (L2 regularization): {self.alpha}")
        print(f"  Input features: {X_train.shape[1]}")

        # Feature selection using F-test
        print(f"\nSelecting top {self.n_features} features...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=self.n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)

        # Get selected feature indices
        self.selected_feature_indices = self.feature_selector.get_support(indices=True)
        print(f"Selected feature indices: {self.selected_feature_indices[:10]}... (showing first 10)")

        # Train Ridge regression
        print(f"\nTraining Ridge regression...")
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.model.fit(X_train_selected, y_train)

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

        # Check for overfitting
        corr_diff = train_metrics['correlation'] - val_metrics['correlation']
        if corr_diff > 0.1:
            print(f"\n⚠️ WARNING: Possible overfitting detected!")
            print(f"  Training correlation ({train_metrics['correlation']:.4f}) >> "
                  f"Validation correlation ({val_metrics['correlation']:.4f})")
            print(f"  Difference: {corr_diff:.4f}")
        else:
            print(f"\n✓ Good generalization (correlation difference: {corr_diff:.4f})")

        print(f"{'='*80}")

        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'selected_features': self.selected_feature_indices
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

        # Select features
        X_selected = self.feature_selector.transform(X)

        # Predict
        predictions = self.model.predict(X_selected)

        return predictions

    def get_feature_importance(self, feature_names: list = None) -> np.ndarray:
        """
        Get feature importance (coefficients).

        Args:
            feature_names: Optional list of feature names

        Returns:
            Array of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get coefficients
        coefficients = self.model.coef_

        # Map back to original feature space
        importance = np.zeros(len(self.selected_feature_indices))
        importance = coefficients

        if feature_names is not None:
            # Create mapping
            selected_names = [feature_names[i] for i in self.selected_feature_indices]
            importance_dict = dict(zip(selected_names, importance))
            # Sort by absolute importance
            sorted_importance = sorted(importance_dict.items(),
                                      key=lambda x: abs(x[1]),
                                      reverse=True)
            return sorted_importance

        return importance

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"LinearModel(name='{self.name}', n_features={self.n_features}, "
                f"alpha={self.alpha}, status={status})")


def demonstrate_linear_model():
    """Demonstration of linear model."""
    print("Linear Model Baseline")
    print("="*80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 200

    # Create features with varying importance
    X = np.random.randn(n_samples, n_features)

    # Create target with linear relationship to top 20 features
    true_coeffs = np.zeros(n_features)
    true_coeffs[:20] = np.random.randn(20) * 0.1  # Top 20 features matter

    y = X @ true_coeffs + np.random.randn(n_samples) * 0.01

    # Split data
    split_idx = int(n_samples * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train model
    model = LinearModel(name="LinearBaseline", n_features=50, alpha=1.0)
    results = model.fit(X_train, y_train, X_val, y_val)

    # Make predictions
    predictions = model.predict(X_val)

    print(f"\nPrediction Examples:")
    print(f"  True: {y_val[:5]}")
    print(f"  Pred: {predictions[:5]}")

    # Feature importance
    print(f"\nTop 10 Most Important Features (by absolute coefficient):")
    importance = model.get_feature_importance()
    sorted_idx = np.argsort(np.abs(importance))[::-1][:10]
    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1}. Feature {model.selected_feature_indices[idx]}: "
              f"{importance[idx]:.6f}")

    print("\n" + "="*80)
    print("Linear model demonstration complete!")


if __name__ == "__main__":
    demonstrate_linear_model()
