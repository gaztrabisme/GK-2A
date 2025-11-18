"""
Data Preparation for Training

Merges all features (spatial, thermal, motion, temporal, PCA components)
and prepares train/test datasets for model training.
"""

import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler


class TrainingDataPreparator:
    def __init__(self):
        self.features_dir = Path("data/processed/features")
        self.pca_dir = Path("pca")

        # Load metadata
        with open('features/metadata.yml', 'r') as f:
            self.feature_metadata = yaml.safe_load(f)

        with open('pca/config/pca_config.yml', 'r') as f:
            self.pca_config = yaml.safe_load(f)

    def load_all_features(self) -> pd.DataFrame:
        """
        Load and merge all feature files into a single DataFrame.

        Returns:
            DataFrame with all features merged on track_id, sequence_id, frame_idx
        """
        print("="*80)
        print("LOADING ALL FEATURES")
        print("="*80)

        # Load spatial features
        spatial_df = pd.read_parquet(self.features_dir / 'spatial_features.parquet')
        print(f"\nSpatial features: {len(spatial_df)} samples")

        # Load thermal features
        thermal_df = pd.read_parquet(self.features_dir / 'thermal_features.parquet')
        print(f"Thermal features: {len(thermal_df)} samples")

        # Merge spatial and thermal (both keyed on filename, timestamp, bbox_id)
        base_df = spatial_df.merge(
            thermal_df,
            on=['filename', 'timestamp', 'bbox_id'],
            how='inner'
        )
        print(f"Merged spatial + thermal: {len(base_df)} samples")

        # Load motion features
        motion_df = pd.read_parquet(self.features_dir / 'motion_features.parquet')
        print(f"Motion features: {len(motion_df)} samples")

        # Load temporal features
        temporal_df = pd.read_parquet(self.features_dir / 'temporal_features.parquet')
        print(f"Temporal features: {len(temporal_df)} samples")

        # We need to connect base_df (has bbox_id) with motion_df (has track_id)
        # Load tracked storms to get bbox_id -> track_id mapping
        with open('data/processed/storm_tracking/tracked_storms.json', 'r') as f:
            import json
            tracks = json.load(f)

        # Build mapping from (sequence_id, frame_idx, bbox_id) to track_id
        # This requires loading sequences to get bbox_ids per frame
        with open('data/processed/sequences/sequences.pkl', 'rb') as f:
            sequences = pickle.load(f)

        print("\nBuilding bbox-to-track mapping...")
        bbox_to_track = {}

        for seq in sequences:
            seq_id = seq['sequence_id']
            for frame_idx, frame in enumerate(seq['frames']):
                for storm in frame['storms']:
                    bbox_id = storm['bbox_id']
                    # Find matching track by position
                    storm_pos = storm['position']

                    # Search for track with this position at this frame
                    for track in tracks:
                        if track['sequence_id'] == seq_id and frame_idx in track['frames']:
                            track_frame_idx = track['frames'].index(frame_idx)
                            track_pos = track['positions'][track_frame_idx]

                            # Check if positions match (should be exact or very close)
                            dist = np.sqrt((storm_pos[0] - track_pos[0])**2 +
                                          (storm_pos[1] - track_pos[1])**2)

                            if dist < 0.001:  # Very close match
                                bbox_to_track[(seq_id, frame_idx, bbox_id)] = track['track_id']
                                break

        print(f"Mapped {len(bbox_to_track)} bboxes to tracks")

        # Add track_id and frame_idx to base_df using the mapping
        # We need to extract sequence_id and frame_idx from timestamp matching
        print("\nMatching base features to tracks...")

        # Add sequence_id by matching timestamps
        base_df['track_id'] = None
        base_df['frame_idx'] = None

        for seq in sequences:
            seq_id = seq['sequence_id']
            for frame_idx, frame in enumerate(seq['frames']):
                frame_ts = frame['timestamp']

                # Find rows with this timestamp
                mask = base_df['timestamp'] == frame_ts

                for idx in base_df[mask].index:
                    bbox_id = base_df.loc[idx, 'bbox_id']
                    key = (seq_id, frame_idx, bbox_id)

                    if key in bbox_to_track:
                        base_df.loc[idx, 'track_id'] = bbox_to_track[key]
                        base_df.loc[idx, 'frame_idx'] = frame_idx
                        base_df.loc[idx, 'sequence_id'] = seq_id

        # Remove rows without track assignment
        base_df = base_df[base_df['track_id'].notna()].copy()
        base_df['track_id'] = base_df['track_id'].astype(int)
        base_df['frame_idx'] = base_df['frame_idx'].astype(int)

        print(f"Matched {len(base_df)} samples to tracks")

        # Now merge with motion and temporal
        merged_df = base_df.merge(
            motion_df,
            on=['track_id', 'sequence_id', 'frame_idx'],
            how='inner'
        )
        print(f"After merging motion: {len(merged_df)} samples")

        merged_df = merged_df.merge(
            temporal_df,
            on=['track_id', 'sequence_id', 'frame_idx'],
            how='inner'
        )
        print(f"After merging temporal: {len(merged_df)} samples")

        print(f"\n✅ Final merged dataset: {len(merged_df)} samples, {len(merged_df.columns)} columns")

        return merged_df

    def apply_pca_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted PCA transformers to thermal and spatial features.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with PCA components added
        """
        print("\n" + "="*80)
        print("APPLYING PCA TRANSFORMATIONS")
        print("="*80)

        df_out = df.copy()

        # Apply thermal PCA
        thermal_config = self.pca_config['pca_results']['thermal_pca']
        thermal_features = thermal_config['feature_names']

        print(f"\nThermal PCA:")
        print(f"  Input features: {thermal_features}")

        # Load scaler and PCA
        with open(self.pca_dir / 'transformers' / 'scaler_thermal.pkl', 'rb') as f:
            scaler_thermal = pickle.load(f)

        with open(self.pca_dir / 'transformers' / 'pca_thermal.pkl', 'rb') as f:
            pca_thermal = pickle.load(f)

        # Transform
        X_thermal = df[thermal_features].fillna(df[thermal_features].mean())
        X_thermal_scaled = scaler_thermal.transform(X_thermal)
        X_thermal_pca = pca_thermal.transform(X_thermal_scaled)

        # Add components to dataframe
        for i, col_name in enumerate(thermal_config['component_names']):
            df_out[col_name] = X_thermal_pca[:, i]

        print(f"  Output components: {thermal_config['component_names']}")

        # Apply spatial PCA
        spatial_config = self.pca_config['pca_results']['spatial_pca']
        spatial_features = spatial_config['feature_names']

        print(f"\nSpatial PCA:")
        print(f"  Input features: {spatial_features}")

        # Load scaler and PCA
        with open(self.pca_dir / 'transformers' / 'scaler_spatial.pkl', 'rb') as f:
            scaler_spatial = pickle.load(f)

        with open(self.pca_dir / 'transformers' / 'pca_spatial.pkl', 'rb') as f:
            pca_spatial = pickle.load(f)

        # Transform
        X_spatial = df[spatial_features].fillna(df[spatial_features].mean())
        X_spatial_scaled = scaler_spatial.transform(X_spatial)
        X_spatial_pca = pca_spatial.transform(X_spatial_scaled)

        # Add components to dataframe
        for i, col_name in enumerate(spatial_config['component_names']):
            df_out[col_name] = X_spatial_pca[:, i]

        print(f"  Output components: {spatial_config['component_names']}")

        print(f"\n✅ PCA components added: {len(df_out.columns) - len(df.columns)} new columns")

        return df_out

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        Get lists of feature columns by category.

        Returns:
            Dictionary mapping category names to column lists
        """
        # Raw features (from metadata)
        spatial_raw = [f['name'] for f in self.feature_metadata['feature_groups']['spatial']['features']]
        thermal_raw = [f['name'] for f in self.feature_metadata['feature_groups']['thermal']['features']]
        motion_raw = [f['name'] for f in self.feature_metadata['feature_groups']['motion']['features']]
        temporal_raw = [f['name'] for f in self.feature_metadata['feature_groups']['temporal']['features']]

        # PCA components (from config)
        thermal_pca = self.pca_config['pca_results']['thermal_pca']['component_names']
        spatial_pca = self.pca_config['pca_results']['spatial_pca']['component_names']

        return {
            'spatial_raw': spatial_raw,
            'thermal_raw': thermal_raw,
            'motion_raw': motion_raw,
            'temporal_raw': temporal_raw,
            'thermal_pca': thermal_pca,
            'spatial_pca': spatial_pca
        }


def main():
    """Test data preparation"""
    prep = TrainingDataPreparator()

    # Load and merge all features
    df = prep.load_all_features()

    # Apply PCA
    df_with_pca = prep.apply_pca_transforms(df)

    # Show available feature columns
    feature_cols = prep.get_feature_columns()

    print("\n" + "="*80)
    print("AVAILABLE FEATURES FOR TRAINING")
    print("="*80)

    for cat_name, cols in feature_cols.items():
        print(f"\n{cat_name} ({len(cols)} features):")
        print(f"  {', '.join(cols)}")

    total_features = sum(len(cols) for cols in feature_cols.values())
    print(f"\nTotal available features: {total_features}")

    print("\n" + "="*80)
    print("Sample data:")
    print(df_with_pca.head())

    print("\n✅ Data preparation test complete!")


if __name__ == "__main__":
    main()
