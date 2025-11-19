"""
Three-way Train/Val/Test Split for Hyperparameter Tuning

Pure sequence-based split to prevent data leakage.
Each split contains completely separate sequences (storms).
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
    Split data into train/val/test by sequences (pure temporal split).

    IMPORTANT: Each split contains completely separate sequences to prevent
    data leakage. With only 8 sequences, this means small splits, but it's
    necessary to ensure models don't learn storm-specific patterns.

    Strategy:
    - With 8 sequences and 3/3/2 split (37.5%/37.5%/25%):
      - Train: First 3 sequences (~270 samples)
      - Val:   Next 3 sequences (~270 samples)
      - Test:  Last 2 sequences (~1424 samples)

    Args:
        df: Full dataframe with 'sequence_id' column
        train_ratio: Fraction for training (default 0.6, but adjusted for discrete sequences)
        val_ratio: Fraction for validation (default 0.2, but adjusted for discrete sequences)
        test_ratio: Fraction for test (default 0.2, but adjusted for discrete sequences)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Get unique sequences sorted by time (temporal ordering)
    sequences = sorted(df['sequence_id'].unique())
    n_sequences = len(sequences)

    print(f"Total sequences: {n_sequences}")

    # For small number of sequences (<=8), use fixed allocation
    # to ensure reasonable validation set
    if n_sequences <= 8:
        # Use 3/3/2 split for 8 sequences (better than 4/1/3)
        n_train = 3
        n_val = 3
        n_test = n_sequences - n_train - n_val

        print(f"⚠️  Small dataset detected ({n_sequences} sequences)")
        print(f"   Using fixed 3/3/{n_test} sequence allocation for better validation")
    else:
        # For larger datasets, use the ratios
        n_train = int(train_ratio * n_sequences)
        n_val = int(val_ratio * n_sequences)
        n_test = n_sequences - n_train - n_val

    # Split sequences temporally (earliest to latest)
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:n_train + n_val]
    test_sequences = sequences[n_train + n_val:]

    # Filter dataframe
    train_df = df[df['sequence_id'].isin(train_sequences)].copy()
    val_df = df[df['sequence_id'].isin(val_sequences)].copy()
    test_df = df[df['sequence_id'].isin(test_sequences)].copy()

    print(f"\n3-way sequence-based split (NO LEAKAGE):")
    print(f"  Train: {len(train_sequences)} seqs ({len(train_df)} samples) - seqs {train_sequences}")
    print(f"  Val:   {len(val_sequences)} seqs ({len(val_df)} samples) - seqs {val_sequences}")
    print(f"  Test:  {len(test_sequences)} seqs ({len(test_df)} samples) - seqs {test_sequences}")

    # Verify no overlap
    overlap_train_val = set(train_sequences) & set(val_sequences)
    overlap_train_test = set(train_sequences) & set(test_sequences)
    overlap_val_test = set(val_sequences) & set(test_sequences)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError("ERROR: Sequence overlap detected between splits!")

    print("  ✓ No sequence overlap - leakage prevented")

    return train_df, val_df, test_df
