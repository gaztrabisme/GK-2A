"""
Three-way Train/Val/Test Split for Hyperparameter Tuning

Hybrid approach:
- Test set: Separate sequences (temporal split, no leakage)
- Train/Val: Sample-based split within training sequences (adequate validation samples)
"""

import pandas as pd
import numpy as np
from typing import Tuple


def three_way_sequence_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    min_val_samples: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test using hybrid approach.

    Strategy:
    1. Reserve test sequences (temporal split, no leakage)
    2. Within remaining sequences, do sample-based train/val split
    3. Ensures adequate validation samples for reliable tuning

    Args:
        df: Full dataframe with 'sequence_id' column
        train_ratio: Fraction for training (default 0.6)
        val_ratio: Fraction for validation (default 0.2)
        test_ratio: Fraction for test (default 0.2)
        min_val_samples: Minimum validation samples (default 100)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Get unique sequences sorted by time
    sequences = sorted(df['sequence_id'].unique())
    n_sequences = len(sequences)

    # Calculate test split (reserve last sequences)
    test_start = int((1 - test_ratio) * n_sequences)

    train_val_sequences = sequences[:test_start]
    test_sequences = sequences[test_start:]

    # Get test data
    test_df = df[df['sequence_id'].isin(test_sequences)].copy()

    # Get train+val data
    train_val_df = df[df['sequence_id'].isin(train_val_sequences)].copy()

    # Now split train_val_df by SAMPLES (not sequences) to ensure adequate validation samples
    # Calculate validation fraction relative to train+val only
    val_fraction = val_ratio / (train_ratio + val_ratio)

    n_train_val = len(train_val_df)
    n_val_desired = int(val_fraction * n_train_val)

    # Ensure we have at least min_val_samples
    n_val = max(n_val_desired, min_val_samples)
    n_val = min(n_val, int(0.4 * n_train_val))  # But not more than 40% of train+val

    # Random sample-based split (with fixed seed for reproducibility)
    np.random.seed(42)
    val_indices = np.random.choice(train_val_df.index, size=n_val, replace=False)

    val_df = train_val_df.loc[val_indices].copy()
    train_df = train_val_df.drop(val_indices).copy()

    print(f"3-way hybrid split:")
    print(f"  Train: {len(train_val_sequences)} seqs, {len(train_df)} samples")
    print(f"  Val:   {len(train_val_sequences)} seqs, {len(val_df)} samples (sample-based)")
    print(f"  Test:  {len(test_sequences)} seqs, {len(test_df)} samples (sequence-based)")

    # Warning if validation is still too small
    if len(val_df) < min_val_samples:
        print(f"  ⚠️  WARNING: Only {len(val_df)} validation samples (minimum {min_val_samples} recommended)")

    return train_df, val_df, test_df
