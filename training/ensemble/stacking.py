"""
Stacking Ensemble for Hurricane Forecasting

Meta-model approach that learns optimal combination of base model predictions.
Uses sequence-based k-fold cross-validation to prevent data leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.base import clone
import lightgbm as lgb


class SequenceKFold:
    """
    K-Fold cross-validator that respects sequence boundaries.

    Each fold contains complete sequences (no sequence is split across folds).
    This prevents data leakage in temporal forecasting.
    """

    def __init__(self, n_splits: int = 3):
        """
        Args:
            n_splits: Number of folds (should match or be less than number of sequences)
        """
        self.n_splits = n_splits

    def split(self, X: np.ndarray, y: np.ndarray = None, sequence_ids: np.ndarray = None):
        """
        Generate train/validation indices for each fold.

        Args:
            X: Feature matrix
            y: Target vector (unused, for sklearn compatibility)
            sequence_ids: Array of sequence IDs for each sample

        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        if sequence_ids is None:
            raise ValueError("sequence_ids must be provided for SequenceKFold")

        # Get unique sequences in temporal order
        unique_sequences = pd.Series(sequence_ids).unique()
        n_sequences = len(unique_sequences)

        if self.n_splits > n_sequences:
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than "
                f"n_sequences={n_sequences}"
            )

        # Assign sequences to folds
        fold_size = n_sequences // self.n_splits

        for fold_idx in range(self.n_splits):
            # Determine which sequences go in validation for this fold
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < self.n_splits - 1 else n_sequences
            val_sequences = unique_sequences[val_start:val_end]

            # Get sample indices for train and validation
            val_mask = np.isin(sequence_ids, val_sequences)
            val_indices = np.where(val_mask)[0]
            train_indices = np.where(~val_mask)[0]

            yield train_indices, val_indices


def generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sequence_ids: np.ndarray,
    base_models: Dict[str, object],
    n_splits: int = 3,
    progress_callback: Optional[callable] = None
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Generate out-of-fold predictions using sequence-based k-fold CV.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        sequence_ids: Sequence ID for each sample (n_samples,)
        base_models: Dict of {model_name: unfitted_model}
        n_splits: Number of folds
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of:
            - oof_predictions: (n_samples, n_models) array of predictions
            - fitted_models: Dict of {model_name: fitted_model} on full training data
    """
    n_samples = X_train.shape[0]
    n_models = len(base_models)
    model_names = list(base_models.keys())

    # Initialize OOF prediction array
    oof_predictions = np.zeros((n_samples, n_models))

    # Sequence-based k-fold
    skf = SequenceKFold(n_splits=n_splits)

    if progress_callback:
        progress_callback(f"Generating out-of-fold predictions ({n_splits} folds)...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train, sequence_ids)):
        if progress_callback:
            progress_callback(
                f"  Fold {fold_idx + 1}/{n_splits}: "
                f"Training on {len(train_idx)} samples, validating on {len(val_idx)}"
            )

        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]

        # Train each base model and generate predictions for this fold
        for model_idx, (model_name, model) in enumerate(base_models.items()):
            # Clone model to avoid modifying original
            fold_model = clone(model)

            # Train on this fold's training data
            fold_model.fit(X_fold_train, y_fold_train)

            # Predict on this fold's validation data
            fold_predictions = fold_model.predict(X_fold_val)

            # Store OOF predictions
            oof_predictions[val_idx, model_idx] = fold_predictions

    # Train final models on full training data
    if progress_callback:
        progress_callback("  Training final base models on full training set...")

    fitted_models = {}
    for model_name, model in base_models.items():
        final_model = clone(model)
        final_model.fit(X_train, y_train)
        fitted_models[model_name] = final_model

    return oof_predictions, fitted_models


class StackingEnsemble:
    """
    Stacking ensemble that combines base model predictions using a meta-model.

    Supports both Ridge regression and LightGBM as meta-models.
    """

    def __init__(
        self,
        base_models: Dict[str, object],
        meta_model_type: str = 'ridge',
        meta_model_params: Optional[Dict] = None,
        n_splits: int = 3
    ):
        """
        Args:
            base_models: Dict of {model_name: unfitted_model}
            meta_model_type: 'ridge' or 'lightgbm'
            meta_model_params: Optional hyperparameters for meta-model
            n_splits: Number of folds for OOF generation
        """
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.meta_model_params = meta_model_params or {}
        self.n_splits = n_splits

        self.fitted_base_models = None
        self.meta_model = None
        self.model_names = list(base_models.keys())

    def _create_meta_model(self):
        """Create meta-model based on type."""
        if self.meta_model_type == 'ridge':
            # Ridge regression with cross-validated alpha
            return Ridge(alpha=1.0, **self.meta_model_params)
        elif self.meta_model_type == 'lightgbm':
            # LightGBM with conservative parameters to avoid overfitting
            default_params = {
                'n_estimators': 100,
                'num_leaves': 15,
                'learning_rate': 0.05,
                'verbosity': -1
            }
            default_params.update(self.meta_model_params)
            return lgb.LGBMRegressor(**default_params)
        else:
            raise ValueError(f"Unknown meta_model_type: {self.meta_model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sequence_ids: np.ndarray,
        progress_callback: Optional[callable] = None
    ):
        """
        Train stacking ensemble.

        Steps:
        1. Generate out-of-fold predictions from base models
        2. Train meta-model on OOF predictions

        Args:
            X_train: Training features
            y_train: Training targets
            sequence_ids: Sequence IDs for k-fold splitting
            progress_callback: Optional progress callback
        """
        if progress_callback:
            progress_callback(
                f"Training stacking ensemble ({self.meta_model_type} meta-model)..."
            )

        # Step 1: Generate OOF predictions
        oof_predictions, self.fitted_base_models = generate_oof_predictions(
            X_train, y_train, sequence_ids, self.base_models,
            n_splits=self.n_splits,
            progress_callback=progress_callback
        )

        # Step 2: Train meta-model on OOF predictions
        if progress_callback:
            progress_callback(f"  Training {self.meta_model_type} meta-model...")

        self.meta_model = self._create_meta_model()
        self.meta_model.fit(oof_predictions, y_train)

        if progress_callback:
            progress_callback("  ✓ Stacking ensemble training complete")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.

        Steps:
        1. Get predictions from each base model
        2. Pass base predictions through meta-model

        Args:
            X_test: Test features

        Returns:
            Final ensemble predictions
        """
        if self.fitted_base_models is None or self.meta_model is None:
            raise ValueError("Ensemble must be trained before prediction")

        # Get predictions from all base models
        base_predictions = np.column_stack([
            model.predict(X_test)
            for model in self.fitted_base_models.values()
        ])

        # Meta-model combines base predictions
        final_predictions = self.meta_model.predict(base_predictions)

        return final_predictions

    def get_weights(self) -> Dict[str, float]:
        """
        Get meta-model weights (coefficients) for each base model.

        Only works for Ridge meta-model (linear). For LightGBM, returns
        feature importances.

        Returns:
            Dict of {model_name: weight}
        """
        if self.meta_model is None:
            raise ValueError("Ensemble must be trained first")

        weights = {}

        if self.meta_model_type == 'ridge':
            # Ridge coefficients directly represent model weights
            coefficients = self.meta_model.coef_
            for i, model_name in enumerate(self.model_names):
                weights[model_name] = float(coefficients[i])

        elif self.meta_model_type == 'lightgbm':
            # LightGBM feature importances (gain-based)
            importances = self.meta_model.feature_importances_
            # Normalize to sum to 1
            total = importances.sum()
            for i, model_name in enumerate(self.model_names):
                weights[model_name] = float(importances[i] / total)

        return weights

    def get_meta_model_info(self) -> Dict:
        """
        Get information about the trained meta-model.

        Returns:
            Dict with meta-model type, weights, and intercept (if applicable)
        """
        if self.meta_model is None:
            raise ValueError("Ensemble must be trained first")

        info = {
            'type': self.meta_model_type,
            'weights': self.get_weights()
        }

        if self.meta_model_type == 'ridge':
            info['intercept'] = float(self.meta_model.intercept_)

        return info
