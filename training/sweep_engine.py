"""
Parameter Sweep Engine

Automated hyperparameter search for finding optimal configurations.
Performs randomized search over split strategies, features, targets, and horizons.
"""

import numpy as np
import pandas as pd
import random
import time
from typing import Dict, List, Any, Callable, Optional
from pathlib import Path

from horizon_data_builder import HorizonDataBuilder
from training_engine import TrainingEngine
from splits.split_strategies import DataSplitter


class SweepEngine:
    """Automated parameter sweep for model optimization"""

    def __init__(self, full_data: pd.DataFrame, feature_cols: Dict[str, List[str]]):
        """
        Initialize sweep engine.

        Args:
            full_data: Full dataset with all features
            feature_cols: Dictionary of feature categories
        """
        self.full_data = full_data
        self.feature_cols = feature_cols
        self.results = []
        self.best_config = None
        self.best_r2 = -np.inf

    def generate_random_config(self, config_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate one random configuration from the config space.

        Args:
            config_space: Dictionary defining parameter ranges

        Returns:
            Random configuration dictionary
        """
        config = {}

        # Split strategy
        if 'split_strategy' in config_space:
            config['split_strategy'] = random.choice(config_space['split_strategy'])

        # Train ratio
        if 'train_ratio' in config_space:
            min_r, max_r = config_space['train_ratio']
            config['train_ratio'] = random.uniform(min_r, max_r)

        # Features
        if 'features' in config_space:
            config['features'] = self._generate_random_features(config_space['features'])

        # Target
        if 'target' in config_space:
            config['target'] = random.choice(config_space['target'])

        # Horizon
        if 'horizon' in config_space:
            config['horizon'] = random.choice(config_space['horizon'])

        # Model
        if 'model' in config_space:
            config['model'] = random.choice(config_space['model'])

        return config

    def _generate_random_features(self, feature_config: Dict) -> List[str]:
        """
        Generate random feature subset based on strategy.

        Args:
            feature_config: Feature generation configuration

        Returns:
            List of selected feature names
        """
        strategy = random.choice(feature_config['strategies'])

        if strategy == 'pca_only':
            # PCA + Motion + Temporal
            features = (
                self.feature_cols['thermal_pca'] +
                self.feature_cols['spatial_pca'] +
                self.feature_cols['motion_raw'] +
                self.feature_cols['temporal_raw']
            )

        elif strategy == 'raw_only':
            # Raw features + Motion + Temporal
            features = (
                self.feature_cols['thermal_raw'] +
                self.feature_cols['spatial_raw'] +
                self.feature_cols['motion_raw'] +
                self.feature_cols['temporal_raw']
            )

        elif strategy == 'motion_temporal_only':
            # Motion + Temporal only (no spatial/thermal)
            features = (
                self.feature_cols['motion_raw'] +
                self.feature_cols['temporal_raw']
            )

        elif strategy == 'random_subset':
            # Random selection from all features
            all_features = []
            for cat in self.feature_cols.values():
                all_features.extend(cat)

            min_feat = feature_config.get('min_features', 5)
            max_feat = feature_config.get('max_features', len(all_features))
            n_features = random.randint(min_feat, min(max_feat, len(all_features)))

            features = random.sample(all_features, n_features)

        else:
            # Default: all features
            features = []
            for cat in self.feature_cols.values():
                features.extend(cat)

        return features

    def train_and_evaluate(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Train and evaluate a single configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary with metrics
        """
        try:
            start_time = time.time()

            # Prepare split
            if config['split_strategy'] == 'sequence':
                train_df, test_df = DataSplitter.sequence_based_temporal_split(
                    self.full_data, config['train_ratio']
                )
            elif config['split_strategy'] == 'date':
                train_df, test_df = DataSplitter.date_based_split(
                    self.full_data, train_ratio=config['train_ratio']
                )
            else:  # random
                train_df, test_df = DataSplitter.random_split(
                    self.full_data, config['train_ratio']
                )

            train_sequences = train_df['sequence_id'].unique().tolist()
            test_sequences = test_df['sequence_id'].unique().tolist()

            # Build horizon dataset
            X, y, idx_curr, idx_fut, df_curr, df_fut = HorizonDataBuilder.build_horizon_dataset(
                self.full_data,
                feature_cols=config['features'],
                target_col=config['target'],
                horizon=config['horizon']
            )

            # Split by sequences
            X_train, y_train, X_test, y_test = HorizonDataBuilder.split_by_sequences(
                df_curr, df_fut, X, y,
                train_sequences, test_sequences
            )

            # Check for sufficient data
            if len(X_test) < 10:
                return {
                    'error': 'Insufficient test samples',
                    'train_r2': np.nan,
                    'test_r2': np.nan,
                    'test_rmse': np.nan,
                    'train_time': 0.0
                }

            # Train model
            engine = TrainingEngine()
            result = engine.train_single_model(
                config['model'],
                X_train, y_train, X_test, y_test,
                feature_names=config['features'],
                model_params=None
            )

            metrics = result['metrics']
            metrics['elapsed_time'] = time.time() - start_time

            return metrics

        except Exception as e:
            return {
                'error': str(e),
                'train_r2': np.nan,
                'test_r2': np.nan,
                'test_rmse': np.nan,
                'train_time': 0.0
            }

    def run_sweep(
        self,
        config_space: Dict[str, Any],
        n_iterations: int,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Run parameter sweep with random search.

        Args:
            config_space: Parameter search space
            n_iterations: Number of random configurations to try
            progress_callback: Optional function(iteration, best_r2, current_result) for progress updates

        Returns:
            DataFrame with all results
        """
        print(f"Starting parameter sweep: {n_iterations} iterations")

        for i in range(n_iterations):
            # Generate random config
            config = self.generate_random_config(config_space)

            # Train and evaluate
            metrics = self.train_and_evaluate(config)

            # Store result
            result = {**config, **metrics}
            result['iteration'] = i
            result['n_features'] = len(config['features'])
            self.results.append(result)

            # Update best
            if not np.isnan(metrics['test_r2']) and metrics['test_r2'] > self.best_r2:
                self.best_r2 = metrics['test_r2']
                self.best_config = config

            # Progress callback
            if progress_callback:
                progress_callback(i, self.best_r2, result)

        return pd.DataFrame(self.results)

    def get_leaderboard(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get top K configurations by test R².

        Args:
            top_k: Number of top results to return

        Returns:
            DataFrame with top configurations
        """
        df = pd.DataFrame(self.results)
        df = df.sort_values('test_r2', ascending=False)
        return df.head(top_k)

    def export_results(self, filepath: str):
        """Export all results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(df)} results to {filepath}")


def test_sweep():
    """Test sweep engine with minimal setup"""
    print("="*80)
    print("TESTING SWEEP ENGINE")
    print("="*80)

    # Create dummy data
    np.random.seed(42)
    n_samples = 500

    data = []
    for track_id in range(10):
        for frame_idx in range(50):
            data.append({
                'track_id': track_id,
                'sequence_id': f'seq_{track_id // 3}',
                'frame_idx': frame_idx,
                'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=frame_idx*10),
                'x_center': 0.5 + 0.001 * frame_idx + np.random.randn() * 0.05,
                'bbox_area': 0.02 + np.random.randn() * 0.002,
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn(),
                'feature_3': np.random.randn()
            })

    df = pd.DataFrame(data)

    feature_cols = {
        'group_a': ['feature_1', 'feature_2'],
        'group_b': ['feature_3']
    }

    # Config space
    config_space = {
        'split_strategy': ['sequence'],
        'train_ratio': [0.7, 0.9],
        'features': {
            'strategies': ['random_subset'],
            'min_features': 2,
            'max_features': 3
        },
        'target': ['x_center'],
        'horizon': [1, 3],
        'model': ['random_forest']
    }

    # Run sweep
    engine = SweepEngine(df, feature_cols)

    def progress(i, best_r2, result):
        if i % 2 == 0:
            print(f"Iteration {i:2d}: Best R²={best_r2:.4f}, Current R²={result.get('test_r2', np.nan):.4f}")

    results_df = engine.run_sweep(config_space, n_iterations=5, progress_callback=progress)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(results_df[['iteration', 'horizon', 'n_features', 'test_r2', 'test_rmse']].to_string())

    print(f"\nBest config: {engine.best_config}")
    print(f"Best R²: {engine.best_r2:.4f}")

    print("\n✅ Sweep engine test complete!")


if __name__ == "__main__":
    test_sweep()
