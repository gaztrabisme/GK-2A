"""
Feature Summary Script

Loads all feature files and prints a comprehensive summary.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_metadata():
    """Load feature metadata"""
    with open('features/metadata.yml', 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata


def summarize_features():
    """Print comprehensive feature summary"""
    print("="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)

    metadata = load_metadata()

    # Load all feature files
    spatial_df = pd.read_parquet('data/processed/features/spatial_features.parquet')
    thermal_df = pd.read_parquet('data/processed/features/thermal_features.parquet')
    motion_df = pd.read_parquet('data/processed/features/motion_features.parquet')
    temporal_df = pd.read_parquet('data/processed/features/temporal_features.parquet')

    print("\n" + "="*80)
    print("FEATURE GROUPS")
    print("="*80)

    for group_name, group_info in metadata['feature_groups'].items():
        print(f"\n{group_name.upper()}:")
        print(f"  Description: {group_info['description']}")
        print(f"  Source: {group_info['source']}")
        print(f"  Min frames required: {group_info['min_frames']}")
        print(f"  Features ({len(group_info['features'])}):")

        for feat in group_info['features']:
            requires_info = f" (requires {feat.get('requires_frames', 1)} frames)" if 'requires_frames' in feat else ""
            print(f"    - {feat['name']}: {feat['description']}{requires_info}")

    print("\n" + "="*80)
    print("FEATURE COUNTS")
    print("="*80)

    print(f"\nSpatial features: {len(spatial_df)} samples, {len(spatial_df.columns)} columns")
    print(f"  Columns: {list(spatial_df.columns)}")

    print(f"\nThermal features: {len(thermal_df)} samples, {len(thermal_df.columns)} columns")
    print(f"  Columns: {list(thermal_df.columns)}")

    print(f"\nMotion features: {len(motion_df)} samples, {len(motion_df.columns)} columns")
    print(f"  Columns: {list(motion_df.columns)}")

    print(f"\nTemporal features: {len(temporal_df)} samples, {len(temporal_df.columns)} columns")
    print(f"  Columns: {list(temporal_df.columns)}")

    # Total unique features
    all_features = set()
    for group_info in metadata['feature_groups'].values():
        for feat in group_info['features']:
            all_features.add(feat['name'])

    print(f"\n" + "="*80)
    print(f"TOTAL UNIQUE FEATURES: {len(all_features)}")
    print("="*80)

    print("\n" + "="*80)
    print("PCA CONFIGURATION")
    print("="*80)

    for pca_name, pca_info in metadata['pca_groups'].items():
        print(f"\n{pca_name}:")
        print(f"  Description: {pca_info['description']}")
        print(f"  Method: {pca_info['method']}")
        print(f"  Variance threshold: {pca_info['variance_threshold']}")
        print(f"  Input features ({len(pca_info['input_features'])}):")
        for feat in pca_info['input_features']:
            print(f"    - {feat}")

    print("\n" + "="*80)
    print("RAW FEATURES (NO PCA)")
    print("="*80)

    for group_name, features in metadata['raw_features'].items():
        print(f"\n{group_name} ({len(features)} features):")
        for feat in features:
            print(f"  - {feat}")

    print("\n" + "="*80)
    print("PREDICTION CONFIGURATION")
    print("="*80)

    print("\nTargets:")
    for target_name, target_info in metadata['prediction_targets'].items():
        print(f"\n  {target_name}:")
        print(f"    Description: {target_info['description']}")
        print(f"    Features: {', '.join(target_info['features'])}")

    print("\nTime Horizons:")
    for horizon in metadata['time_horizons']:
        print(f"  - {horizon['name']}: {horizon['frames_ahead']} frames ({horizon['real_time']}) - {horizon['use_case']}")

    print(f"\nLookback Window: {metadata['lookback']['default_frames']} frames ({metadata['lookback']['default_duration']})")

    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    stats = metadata['dataset_statistics']
    print(f"\nTotal images: {stats['total_images']}")
    print(f"Total bounding boxes: {stats['total_bboxes']}")
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total storm tracks: {stats['total_tracks']}")
    print(f"Longest track: {stats['longest_track_frames']} frames")

    print("\nFeature samples:")
    for feat_group, feat_stats in stats['features'].items():
        print(f"  {feat_group}:")
        for stat_name, stat_val in feat_stats.items():
            print(f"    {stat_name}: {stat_val}")

    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Implement PCA analysis with elbow detection")
    print("  2. Build training GUI (Gradio)")
    print("  3. Train models (RF, XGBoost, LightGBM)")
    print("="*80)


if __name__ == "__main__":
    summarize_features()
