"""
Evaluation Metrics for Hurricane Forecasting

Computes RMSE, MAE, R² and other regression metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any
import time


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'valid_samples': 0
        }

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'r2': r2_score(y_true_clean, y_pred_clean),
        'valid_samples': len(y_true_clean)
    }

    return metrics


def evaluate_model(model, X_train, y_train, X_test, y_test, train_time: float = 0.0) -> Dict[str, Any]:
    """
    Evaluate a trained model on train and test sets.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        train_time: Training time in seconds

    Returns:
        Dictionary with train and test metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    # Combine results
    results = {
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'train_time': train_time,
        'train_samples': len(y_train),
        'test_samples': len(y_test)
    }

    return results


def format_metrics_table(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Format metrics as a readable table.

    Args:
        results: Dictionary mapping model_name to metrics dict

    Returns:
        Formatted string table
    """
    # Header
    header = f"{'Model':<20} {'Train Time':<12} {'Train RMSE':<12} {'Test RMSE':<12} {'Train MAE':<12} {'Test MAE':<12} {'Train R²':<10} {'Test R²':<10}"
    separator = "="*120

    lines = [separator, header, separator]

    # Rows
    for model_name, metrics in results.items():
        row = (
            f"{model_name:<20} "
            f"{metrics['train_time']:>10.2f}s "
            f"{metrics['train_rmse']:>11.4f} "
            f"{metrics['test_rmse']:>11.4f} "
            f"{metrics['train_mae']:>11.4f} "
            f"{metrics['test_mae']:>11.4f} "
            f"{metrics['train_r2']:>9.4f} "
            f"{metrics['test_r2']:>9.4f}"
        )
        lines.append(row)

    lines.append(separator)

    return "\n".join(lines)


def test_metrics():
    """Test metrics calculation"""
    print("="*80)
    print("TESTING EVALUATION METRICS")
    print("="*80)

    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1  # Add some noise

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Test with NaN values
    y_true_nan = y_true.copy()
    y_true_nan[::10] = np.nan

    metrics_nan = calculate_metrics(y_true_nan, y_pred)

    print("\nMetrics with NaN handling:")
    for name, value in metrics_nan.items():
        if name == 'valid_samples':
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: {value:.4f}")

    print("\n✅ Metrics test complete!")


if __name__ == "__main__":
    test_metrics()
