"""
Training Engine

Coordinates model training across multiple models and time horizons.
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional
import sys
sys.path.append(str(Path(__file__).parent))

from models.tree_models import TreeModelFactory, get_feature_importance
from splits.split_strategies import DataSplitter
from ensemble import VotingEnsemble
from ensemble.stacking import StackingEnsemble
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.metrics import evaluate_model, calculate_metrics


class TrainingEngine:
    """Orchestrates multi-model, multi-horizon training"""

    def __init__(self, output_dir: str = "training/trained_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        model_params: Optional[Dict] = None
    ) -> Dict:
        """
        Train and evaluate a single model.

        Args:
            model_name: 'random_forest', 'xgboost', or 'lightgbm'
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: List of feature names
            model_params: Optional custom hyperparameters

        Returns:
            Dictionary with model, metrics, and metadata
        """
        # Create model
        model = TreeModelFactory.create_model(model_name, model_params)

        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, train_time)

        # Feature importance
        feature_importance = get_feature_importance(model, feature_names)

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_name': model_name,
            'feature_names': feature_names
        }

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        model_names: List[str],
        target_name: str = "target",
        progress_callback: Optional[Callable] = None,
        enable_ensemble: bool = True,
        model_params_dict: Optional[Dict[str, Dict]] = None,
        ensemble_type: str = 'voting',
        meta_model_type: str = 'ridge',
        sequence_ids_train: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train multiple models and compare results.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: List of feature names
            model_names: List of model names to train
            target_name: Name of target variable (for saving)
            progress_callback: Optional function(message) to report progress
            enable_ensemble: If True, create ensemble from trained models
            model_params_dict: Optional dict mapping model names to hyperparameters
                              e.g., {'lightgbm': {'n_estimators': 300, 'num_leaves': 50}}
            ensemble_type: 'voting' for simple averaging, 'stacking' for meta-model
            meta_model_type: 'ridge' or 'lightgbm' (only used if ensemble_type='stacking')
            sequence_ids_train: Sequence IDs for training data (required for stacking)

        Returns:
            Dictionary mapping model names to results (includes 'ensemble'/'stacking' if enabled)
        """
        results = {}
        trained_models = {}

        for i, model_name in enumerate(model_names):
            if progress_callback:
                progress_callback(f"Training {model_name} ({i+1}/{len(model_names)})...")

            try:
                # Get hyperparameters for this model if available
                model_params = model_params_dict.get(model_name) if model_params_dict else None

                if model_params:
                    progress_callback(f"  Using tuned hyperparameters for {model_name}")

                result = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test, feature_names,
                    model_params=model_params
                )
                results[model_name] = result
                trained_models[model_name] = result['model']

                if progress_callback:
                    progress_callback(
                        f"✓ {model_name}: Test RMSE={result['metrics']['test_rmse']:.4f}, "
                        f"R²={result['metrics']['test_r2']:.4f}"
                    )

            except Exception as e:
                if progress_callback:
                    progress_callback(f"✗ {model_name} failed: {str(e)}")
                results[model_name] = {'error': str(e)}

        # Create ensemble if enabled and we have multiple successful models
        if enable_ensemble and len(trained_models) >= 2:
            if ensemble_type == 'voting':
                # Simple voting ensemble (averaging predictions)
                if progress_callback:
                    progress_callback(f"Creating voting ensemble from {len(trained_models)} models...")

                try:
                    ensemble = VotingEnsemble(trained_models)
                    ensemble_metrics = ensemble.evaluate(X_train, y_train, X_test, y_test)

                    results['ensemble'] = {
                        'model': ensemble,
                        'metrics': ensemble_metrics,
                        'train_time': 0.0  # Ensemble uses pre-trained models
                    }

                    if progress_callback:
                        progress_callback(
                            f"✓ Voting Ensemble: Test RMSE={ensemble_metrics['test_rmse']:.4f}, "
                            f"R²={ensemble_metrics['test_r2']:.4f}"
                        )

                except Exception as e:
                    if progress_callback:
                        progress_callback(f"✗ Voting ensemble failed: {str(e)}")
                    results['ensemble'] = {'error': str(e)}

            elif ensemble_type == 'stacking':
                # Stacking ensemble (meta-model learns combination)
                if sequence_ids_train is None:
                    error_msg = "sequence_ids_train required for stacking ensemble"
                    if progress_callback:
                        progress_callback(f"✗ Stacking failed: {error_msg}")
                    results['stacking'] = {'error': error_msg}
                    return results

                if progress_callback:
                    progress_callback(
                        f"Creating stacking ensemble ({meta_model_type} meta-model) "
                        f"from {len(trained_models)} models..."
                    )

                try:
                    # Create unfitted base models for stacking
                    base_models = {}
                    for model_name in trained_models.keys():
                        model_params = model_params_dict.get(model_name) if model_params_dict else None
                        base_models[model_name] = TreeModelFactory.create_model(model_name, model_params)

                    # Create and train stacking ensemble
                    stacking = StackingEnsemble(
                        base_models=base_models,
                        meta_model_type=meta_model_type,
                        n_splits=3  # Use 3-fold CV for OOF predictions
                    )

                    start_time = time.time()
                    stacking.train(X_train, y_train, sequence_ids_train, progress_callback=progress_callback)
                    stacking_train_time = time.time() - start_time

                    # Evaluate stacking ensemble
                    y_train_pred = stacking.predict(X_train)
                    y_test_pred = stacking.predict(X_test)

                    from evaluation.metrics import calculate_metrics
                    train_metrics = calculate_metrics(y_train, y_train_pred)
                    test_metrics = calculate_metrics(y_test, y_test_pred)

                    stacking_metrics = {
                        'train_rmse': train_metrics['rmse'],
                        'train_mae': train_metrics['mae'],
                        'train_r2': train_metrics['r2'],
                        'test_rmse': test_metrics['rmse'],
                        'test_mae': test_metrics['mae'],
                        'test_r2': test_metrics['r2'],
                        'train_time': stacking_train_time
                    }

                    # Get meta-model weights
                    meta_info = stacking.get_meta_model_info()

                    # Safety check: Compare stacking vs best base model
                    best_base_r2 = max(results[name]['metrics']['test_r2']
                                      for name in trained_models.keys())
                    best_base_name = max(trained_models.keys(),
                                        key=lambda n: results[n]['metrics']['test_r2'])

                    stacking_r2 = stacking_metrics['test_r2']

                    # If stacking is significantly worse, use best base model instead
                    if stacking_r2 < best_base_r2 - 0.05:  # 5% threshold
                        if progress_callback:
                            progress_callback(
                                f"⚠️  Stacking R²={stacking_r2:.4f} < Best model ({best_base_name}) R²={best_base_r2:.4f}"
                            )
                            progress_callback(f"  → Using {best_base_name} instead of stacking (likely insufficient training data)")

                        # Use best base model's results instead
                        results['stacking'] = {
                            'model': results[best_base_name]['model'],
                            'metrics': results[best_base_name]['metrics'],
                            'fallback_used': True,
                            'fallback_from': best_base_name,
                            'stacking_r2_failed': stacking_r2,
                            'train_time': results[best_base_name]['metrics']['train_time']
                        }
                    else:
                        # Stacking is good, use it
                        results['stacking'] = {
                            'model': stacking,
                            'metrics': stacking_metrics,
                            'meta_model_info': meta_info,
                            'train_time': stacking_train_time
                        }

                        if progress_callback:
                            progress_callback(
                                f"✓ Stacking ({meta_model_type}): Test RMSE={stacking_metrics['test_rmse']:.4f}, "
                                f"R²={stacking_metrics['test_r2']:.4f}"
                            )
                            progress_callback(f"  Meta-model weights: {meta_info['weights']}")

                except Exception as e:
                    if progress_callback:
                        progress_callback(f"✗ Stacking ensemble failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    results['stacking'] = {'error': str(e)}

        return results

    def save_model(self, model, model_name: str, target_name: str, horizon: Optional[int] = None):
        """
        Save trained model to disk.

        Args:
            model: Trained model
            model_name: Name of model type
            target_name: Target variable name
            horizon: Optional time horizon (t+N)
        """
        if horizon:
            filename = f"{model_name}_t{horizon}_{target_name}.pkl"
        else:
            filename = f"{model_name}_{target_name}.pkl"

        filepath = self.output_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        return filepath

    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def test_engine():
    """Test training engine with dummy data"""
    print("="*80)
    print("TESTING TRAINING ENGINE")
    print("="*80)

    # Create dummy data
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(n_samples) * 0.1

    # Split
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    feature_names = [f"feature_{i}" for i in range(n_features)]

    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Create engine
    engine = TrainingEngine()

    # Progress callback
    def print_progress(msg):
        print(f"  {msg}")

    # Train models (only RF for now since we don't have xgboost/lightgbm installed)
    print("\nTraining models...")
    results = engine.train_all_models(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        model_names=['random_forest'],
        target_name="test_target",
        progress_callback=print_progress
    )

    # Show results
    print("\nResults:")
    for model_name, result in results.items():
        if 'error' in result:
            print(f"\n{model_name}: ERROR - {result['error']}")
        else:
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
            print(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
            print(f"  Train R²:   {metrics['train_r2']:.4f}")
            print(f"  Test R²:    {metrics['test_r2']:.4f}")
            print(f"  Train time: {metrics['train_time']:.2f}s")

            # Top 3 important features
            if result['feature_importance']:
                sorted_importance = sorted(
                    result['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                print(f"  Top features: {', '.join([f'{name}({imp:.3f})' for name, imp in sorted_importance])}")

    print("\n✅ Training engine test complete!")


if __name__ == "__main__":
    test_engine()
