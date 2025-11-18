"""
Horizon Data Builder

Creates time-shifted datasets for multi-horizon forecasting.
For horizon t+N, creates pairs of (features_at_t, target_at_t+N).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class HorizonDataBuilder:
    """Builds datasets for different prediction horizons"""

    @staticmethod
    def build_horizon_dataset(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        horizon: int,
        lookback: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Build dataset for a specific prediction horizon.

        For horizon=6, this creates pairs where:
        - X: features at frame t
        - y: target at frame t+6 (6 frames into future)

        Args:
            df: Full dataframe with sequence_id, track_id, frame_idx
            feature_cols: List of feature column names
            target_col: Target column name
            horizon: Number of frames ahead to predict (e.g., 6 for t+6)
            lookback: Number of historical frames to include (future feature)

        Returns:
            Tuple of (X_current, y_future, indices_current, indices_future, df_current, df_future)
        """
        print(f"\nBuilding dataset for horizon t+{horizon}...")
        print(f"  Target: {target_col}")
        print(f"  Features: {len(feature_cols)} columns")

        valid_pairs = []

        # Group by track (each storm's time series)
        for track_id in df['track_id'].unique():
            track_df = df[df['track_id'] == track_id].sort_values('frame_idx').reset_index(drop=False)

            # For each frame in this track
            for i in range(len(track_df)):
                current_frame = track_df.iloc[i]
                current_idx = current_frame['index']  # original df index
                current_frame_num = current_frame['frame_idx']

                # Look for target frame (horizon frames ahead)
                target_frame_num = current_frame_num + horizon

                # Check if target frame exists in same track
                target_rows = track_df[track_df['frame_idx'] == target_frame_num]

                if len(target_rows) == 1:
                    target_idx = target_rows.iloc[0]['index']

                    # Verify they're in same sequence (important!)
                    if current_frame['sequence_id'] == target_rows.iloc[0]['sequence_id']:
                        valid_pairs.append({
                            'current_idx': current_idx,
                            'target_idx': target_idx,
                            'track_id': track_id,
                            'current_frame': current_frame_num,
                            'target_frame': target_frame_num
                        })

        print(f"  Found {len(valid_pairs)} valid pairs (frames with t+{horizon} target)")

        if len(valid_pairs) == 0:
            raise ValueError(f"No valid pairs found for horizon t+{horizon}. Dataset too short?")

        # Extract indices
        current_indices = [p['current_idx'] for p in valid_pairs]
        target_indices = [p['target_idx'] for p in valid_pairs]

        # Build X (features at current time)
        df_current = df.loc[current_indices].copy()
        X_current = df_current[feature_cols].fillna(df_current[feature_cols].mean()).values

        # Build y (target at future time)
        df_future = df.loc[target_indices].copy()
        y_future = df_future[target_col].values

        print(f"  X shape: {X_current.shape}")
        print(f"  y shape: {y_future.shape}")
        print(f"  Valid samples: {(~np.isnan(y_future)).sum()}")

        return X_current, y_future, np.array(current_indices), np.array(target_indices), df_current, df_future

    @staticmethod
    def split_by_sequences(
        df_current: pd.DataFrame,
        df_future: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        train_sequences: List[str],
        test_sequences: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split horizon dataset into train/test by sequences.

        Args:
            df_current: DataFrame of current frames
            df_future: DataFrame of future frames
            X: Feature matrix
            y: Target vector
            train_sequences: List of sequence IDs for training
            test_sequences: List of sequence IDs for testing

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Use current frame's sequence for split decision
        train_mask = df_current['sequence_id'].isin(train_sequences).values
        test_mask = df_current['sequence_id'].isin(test_sequences).values

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        print(f"\n  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")

        return X_train, y_train, X_test, y_test


def test_horizon_builder():
    """Test horizon data builder"""
    print("="*80)
    print("TESTING HORIZON DATA BUILDER")
    print("="*80)

    # Create dummy sequential data
    data = []
    for track_id in range(3):
        for frame_idx in range(20):
            data.append({
                'track_id': track_id,
                'sequence_id': 'seq_001',
                'frame_idx': frame_idx,
                'x_center': 0.5 + 0.01 * frame_idx + np.random.randn() * 0.01,
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn()
            })

    df = pd.DataFrame(data)
    print(f"\nTest data: {len(df)} samples, {df['track_id'].nunique()} tracks")

    # Test different horizons
    for horizon in [1, 3, 6]:
        try:
            X, y, idx_curr, idx_fut, df_curr, df_fut = HorizonDataBuilder.build_horizon_dataset(
                df,
                feature_cols=['feature_1', 'feature_2'],
                target_col='x_center',
                horizon=horizon
            )

            print(f"\nHorizon t+{horizon}:")
            print(f"  Samples: {len(y)}")
            print(f"  First target frame: {df_fut.iloc[0]['frame_idx']} (from current frame {df_curr.iloc[0]['frame_idx']})")
            print(f"  Last target frame: {df_fut.iloc[-1]['frame_idx']} (from current frame {df_curr.iloc[-1]['frame_idx']})")

        except ValueError as e:
            print(f"\nHorizon t+{horizon}: {e}")

    print("\n✅ Horizon builder test complete!")


if __name__ == "__main__":
    test_horizon_builder()
