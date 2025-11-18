"""
Train/Test Split Strategies

Implements various splitting approaches respecting temporal structure.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime


class DataSplitter:
    """Handles various train/test split strategies"""

    @staticmethod
    def sequence_based_temporal_split(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by sequences, respecting temporal order.

        Args:
            df: DataFrame with 'sequence_id' column
            train_ratio: Proportion of sequences for training
            random_seed: Random seed (not used for temporal split, but kept for consistency)

        Returns:
            Tuple of (train_df, test_df)
        """
        # Get unique sequences sorted by first timestamp
        sequences = df.groupby('sequence_id')['timestamp'].min().sort_values()
        sequence_ids = sequences.index.tolist()

        # Split sequences
        split_idx = int(len(sequence_ids) * train_ratio)
        train_sequences = sequence_ids[:split_idx]
        test_sequences = sequence_ids[split_idx:]

        # Split data
        train_df = df[df['sequence_id'].isin(train_sequences)].copy()
        test_df = df[df['sequence_id'].isin(test_sequences)].copy()

        return train_df, test_df

    @staticmethod
    def date_based_split(
        df: pd.DataFrame,
        cutoff_date: str = None,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by a date cutoff.

        Args:
            df: DataFrame with 'timestamp' column
            cutoff_date: Date cutoff (ISO format). If None, computed from train_ratio
            train_ratio: Used if cutoff_date is None

        Returns:
            Tuple of (train_df, test_df)
        """
        if cutoff_date is None:
            # Compute cutoff from ratio
            sorted_dates = df['timestamp'].sort_values()
            cutoff_idx = int(len(sorted_dates) * train_ratio)
            cutoff = sorted_dates.iloc[cutoff_idx]
        else:
            cutoff = pd.to_datetime(cutoff_date)

        train_df = df[df['timestamp'] < cutoff].copy()
        test_df = df[df['timestamp'] >= cutoff].copy()

        return train_df, test_df

    @staticmethod
    def random_split(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Random split (not recommended for time series, but included for comparison).

        Args:
            df: DataFrame
            train_ratio: Proportion for training
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df)
        """
        np.random.seed(random_seed)

        # Shuffle indices
        indices = np.arange(len(df))
        np.random.shuffle(indices)

        # Split
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()

        return train_df, test_df

    @staticmethod
    def get_split_info(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Get information about a train/test split.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame

        Returns:
            Dictionary with split statistics
        """
        info = {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_ratio': len(train_df) / (len(train_df) + len(test_df)),
            'total_samples': len(train_df) + len(test_df)
        }

        # Add temporal info if available
        if 'timestamp' in train_df.columns:
            info['train_date_range'] = (
                train_df['timestamp'].min().isoformat(),
                train_df['timestamp'].max().isoformat()
            )
            info['test_date_range'] = (
                test_df['timestamp'].min().isoformat(),
                test_df['timestamp'].max().isoformat()
            )

        # Add sequence info if available
        if 'sequence_id' in train_df.columns:
            info['train_sequences'] = train_df['sequence_id'].nunique()
            info['test_sequences'] = test_df['sequence_id'].nunique()

        # Add track info if available
        if 'track_id' in train_df.columns:
            info['train_tracks'] = train_df['track_id'].nunique()
            info['test_tracks'] = test_df['track_id'].nunique()

        return info


def test_splits():
    """Test split strategies"""
    print("="*80)
    print("TESTING SPLIT STRATEGIES")
    print("="*80)

    # Create dummy data
    dates = pd.date_range('2023-10-15', periods=100, freq='10min')
    df = pd.DataFrame({
        'timestamp': dates,
        'sequence_id': ['seq_1'] * 40 + ['seq_2'] * 30 + ['seq_3'] * 30,
        'track_id': np.random.randint(0, 10, 100),
        'feature': np.random.randn(100)
    })

    print(f"\nTest data: {len(df)} samples, {df['sequence_id'].nunique()} sequences")

    # Test each strategy
    strategies = {
        'Sequence-based Temporal': DataSplitter.sequence_based_temporal_split,
        'Date-based': DataSplitter.date_based_split,
        'Random': DataSplitter.random_split
    }

    for name, strategy in strategies.items():
        print(f"\n{name} Split:")
        train_df, test_df = strategy(df, train_ratio=0.8)

        info = DataSplitter.get_split_info(train_df, test_df)
        print(f"  Train: {info['train_samples']} samples, {info.get('train_sequences', 'N/A')} sequences")
        print(f"  Test: {info['test_samples']} samples, {info.get('test_sequences', 'N/A')} sequences")
        print(f"  Ratio: {info['train_ratio']:.2%}")

    print("\n✅ Split strategies test complete!")


if __name__ == "__main__":
    test_splits()
