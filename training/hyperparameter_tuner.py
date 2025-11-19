"""
Bayesian Hyperparameter Tuning for Hurricane Forecasting Models

Uses Optuna for intelligent hyperparameter optimization.
"""

import numpy as np
import optuna
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import r2_score, mean_squared_error
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.tree_models import TreeModelFactory


class HyperparameterTuner:
    """Bayesian hyperparameter optimization using Optuna"""

    def __init__(self, model_name: str):
        """
        Initialize tuner for a specific model.

        Args:
            model_name: Name of model to tune ('random_forest', 'xgboost', 'lightgbm')
        """
        self.model_name = model_name
        self.study = None
        self.best_params = None

    @staticmethod
    def get_search_space(model_name: str) -> Dict:
        """
        Get hyperparameter search space for a given model.

        Args:
            model_name: Model name

        Returns:
            Dictionary describing search space
        """
        if model_name == 'random_forest':
            return {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
                'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                'min_samples_split': {'type': 'int', 'low': 5, 'high': 50},
                'min_samples_leaf': {'type': 'int', 'low': 2, 'high': 20},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.5, 0.7]}
            }

        elif model_name == 'xgboost':
            return {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 10.0}
            }

        elif model_name == 'lightgbm':
            return {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
                'num_leaves': {'type': 'int', 'low': 15, 'high': 127},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'min_child_samples': {'type': 'int', 'low': 10, 'high': 50},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 10.0}
            }

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def create_objective(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Callable:
        """
        Create objective function for Optuna to optimize.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Objective function that takes trial and returns metric to maximize
        """
        search_space = self.get_search_space(self.model_name)

        def objective(trial):
            """Objective function to maximize validation R²"""

            # Suggest hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_config, param_config['choices']
                    )

            # Train model with these hyperparameters
            try:
                model = TreeModelFactory.create_model(self.model_name, params)
                model.fit(X_train, y_train)

                # Evaluate on validation set
                y_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, y_pred)

                return val_r2

            except Exception as e:
                # Return very poor score if training fails
                print(f"Trial failed: {str(e)}")
                return -999.0

        return objective

    def tune(self,
             X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             n_trials: int = 50,
             progress_callback: Optional[Callable] = None) -> Dict:
        """
        Run hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of trials to run
            progress_callback: Optional callback(trial_num, best_r2, params)

        Returns:
            Dictionary with best parameters and performance
        """
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER TUNING: {self.model_name}")
        print(f"{'='*80}")
        print(f"Trials: {n_trials}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print()

        # Create objective function
        objective = self.create_objective(X_train, y_train, X_val, y_val)

        # Create study
        self.study = optuna.create_study(
            direction='maximize',  # Maximize R²
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Add callback for progress reporting
        def optuna_callback(study, trial):
            if progress_callback:
                progress_callback(trial.number + 1, study.best_value, study.best_params)

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[optuna_callback] if progress_callback else None,
            show_progress_bar=True
        )

        # Store best parameters
        self.best_params = self.study.best_params

        print(f"\n{'='*80}")
        print("TUNING COMPLETE")
        print(f"{'='*80}")
        print(f"Best validation R²: {self.study.best_value:.4f}")
        print(f"Best parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"{'='*80}\n")

        return {
            'best_params': self.best_params,
            'best_r2': self.study.best_value,
            'n_trials': n_trials,
            'study': self.study
        }

    def get_best_model(self):
        """
        Get a model instance with the best hyperparameters.

        Returns:
            Untrained model with best parameters
        """
        if self.best_params is None:
            raise ValueError("Must run tune() first")

        return TreeModelFactory.create_model(self.model_name, self.best_params)


def test_tuner():
    """Test hyperparameter tuner with dummy data"""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    print("="*80)
    print("TESTING HYPERPARAMETER TUNER")
    print("="*80)

    # Create dummy data
    X, y = make_regression(n_samples=500, n_features=6, noise=10, random_state=42)

    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Test with LightGBM
    tuner = HyperparameterTuner('lightgbm')

    def progress(trial_num, best_r2, params):
        if trial_num % 10 == 0:
            print(f"Trial {trial_num}: Best R² = {best_r2:.4f}")

    result = tuner.tune(X_train, y_train, X_val, y_val, n_trials=20, progress_callback=progress)

    # Train final model with best params
    best_model = tuner.get_best_model()
    best_model.fit(X_train, y_train)

    # Evaluate
    test_r2 = r2_score(y_test, best_model.predict(X_test))
    print(f"\n✅ Test R² with tuned params: {test_r2:.4f}")


if __name__ == "__main__":
    test_tuner()
