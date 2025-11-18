"""
Tree-based Models for Hurricane Forecasting

Implements Random Forest, XGBoost, and LightGBM with sensible defaults.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from typing import Dict, Any


class TreeModelFactory:
    """Factory for creating tree-based regression models"""

    @staticmethod
    def get_default_params(model_name: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for each model.

        Args:
            model_name: 'random_forest', 'xgboost', or 'lightgbm'

        Returns:
            Dictionary of hyperparameters
        """
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'n_jobs': -1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'n_jobs': -1,
                'random_state': 42,
                'verbose': -1
            }
        }

        return defaults.get(model_name, {})

    @staticmethod
    def create_model(model_name: str, custom_params: Dict[str, Any] = None):
        """
        Create a model instance with specified parameters.

        Args:
            model_name: 'random_forest', 'xgboost', or 'lightgbm'
            custom_params: Optional custom hyperparameters (overrides defaults)

        Returns:
            Instantiated model object
        """
        # Get default params
        params = TreeModelFactory.get_default_params(model_name)

        # Override with custom params if provided
        if custom_params:
            params.update(custom_params)

        # Create model
        if model_name == 'random_forest':
            return RandomForestRegressor(**params)
        elif model_name == 'xgboost':
            return XGBRegressor(**params)
        elif model_name == 'lightgbm':
            return LGBMRegressor(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose from: random_forest, xgboost, lightgbm")


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        Dictionary mapping feature names to importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    else:
        return {}


def test_models():
    """Test model creation"""
    print("="*80)
    print("TESTING TREE MODEL FACTORY")
    print("="*80)

    models = ['random_forest', 'xgboost', 'lightgbm']

    for model_name in models:
        print(f"\n{model_name.upper()}:")

        # Create model
        model = TreeModelFactory.create_model(model_name)
        print(f"  Created: {type(model).__name__}")

        # Show params
        params = TreeModelFactory.get_default_params(model_name)
        print(f"  Default params:")
        for key, val in list(params.items())[:5]:
            print(f"    {key}: {val}")
        print(f"    ... ({len(params)} total params)")

    print("\n✅ Model factory test complete!")


if __name__ == "__main__":
    test_models()
