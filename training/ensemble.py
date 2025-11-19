"""
Ensemble Methods for Hurricane Forecasting

Simple voting ensemble that averages predictions from multiple models.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class VotingEnsemble:
    """Simple voting ensemble that averages predictions from multiple models"""

    def __init__(self, models: Dict[str, Any]):
        """
        Initialize voting ensemble.

        Args:
            models: Dictionary of {model_name: trained_model}
                   e.g., {'random_forest': rf_model, 'xgboost': xgb_model, 'lightgbm': lgbm_model}
        """
        self.models = models
        self.model_names = list(models.keys())

        print(f"Voting Ensemble initialized with {len(models)} models: {self.model_names}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by averaging all model predictions.

        Args:
            X: Feature matrix

        Returns:
            Averaged predictions
        """
        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)

        # Simple average
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble on train and test sets.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with metrics
        """
        # Train predictions
        train_pred = self.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)

        # Test predictions
        test_pred = self.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)

        return {
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'train_time': 0.0  # Ensemble uses pre-trained models
        }
