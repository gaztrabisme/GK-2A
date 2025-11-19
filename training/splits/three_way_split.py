"""
Three-way Train/Val/Test Split for Hyperparameter Tuning

Splits data into train/validation/test while respecting sequence boundaries.
"""

import pandas as pd
from typing import Tuple


def three_way_sequence_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test by sequences (temporal split).

    Args:
        df: Full dataframe with 'sequence_id' column
        train_ratio: Fraction for training (default 0.6)
        val_ratio: Fraction for validation (default 0.2)
        test_ratio: Fraction for test (default 0.2)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Get unique sequences sorted by time
    sequences = sorted(df['sequence_id'].unique())
    n_sequences = len(sequences)

    # Calculate split indices
    train_end = int(train_ratio * n_sequences)
    val_end = train_end + int(val_ratio * n_sequences)

    # Split sequences
    train_sequences = sequences[:train_end]
    val_sequences = sequences[train_end:val_end]
    test_sequences = sequences[val_end:]

    # Filter dataframe
    train_df = df[df['sequence_id'].isin(train_sequences)].copy()
    val_df = df[df['sequence_id'].isin(val_sequences)].copy()
    test_df = df[df['sequence_id'].isin(test_sequences)].copy()

    print(f"3-way split:")
    print(f"  Train: {len(train_sequences)} seqs ({len(train_df)} samples)")
    print(f"  Val:   {len(val_sequences)} seqs ({len(val_df)} samples)")
    print(f"  Test:  {len(test_sequences)} seqs ({len(test_df)} samples)")

    return train_df, val_df, test_df
