"""
Build temporal sequences from spatial and thermal features.

Detects temporal gaps and creates continuous sequences (≤10 min gaps).
Provides the central data structure for feature engineering.
"""

import pickle
import json
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm


class SequenceBuilder:
    def __init__(self,
                 spatial_features_path="data/processed/features/spatial_features.parquet",
                 thermal_features_path="data/processed/features/thermal_features.parquet",
                 gap_threshold_minutes=10):
        self.spatial_features_path = Path(spatial_features_path)
        self.thermal_features_path = Path(thermal_features_path)
        self.gap_threshold_minutes = gap_threshold_minutes

    def load_features(self):
        """Load spatial and thermal features"""
        print("Loading features...")

        df_spatial = pd.read_parquet(self.spatial_features_path)
        df_thermal = pd.read_parquet(self.thermal_features_path)

        # Merge on filename, timestamp, bbox_id
        df = pd.merge(
            df_spatial,
            df_thermal,
            on=['filename', 'timestamp', 'bbox_id'],
            how='inner'
        )

        # Sort by timestamp
        df = df.sort_values(['timestamp', 'bbox_id']).reset_index(drop=True)

        print(f"Loaded {len(df)} bounding boxes")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def detect_sequences(self, df):
        """Detect continuous sequences based on temporal gaps"""
        print(f"\nDetecting sequences (gap threshold: {self.gap_threshold_minutes} minutes)...")

        # Get unique timestamps
        timestamps = df['timestamp'].unique()
        timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)

        # Calculate gaps
        gaps = timestamps.diff()

        # Identify sequence breaks (gaps > threshold)
        threshold = timedelta(minutes=self.gap_threshold_minutes)
        breaks = (gaps > threshold) | gaps.isna()

        # Assign sequence IDs
        sequence_ids = breaks.cumsum()

        # Create timestamp to sequence mapping
        timestamp_to_seq = pd.Series(
            sequence_ids.values,
            index=timestamps.values,
            name='sequence_id'
        )

        # Map back to dataframe
        df['sequence_id'] = df['timestamp'].map(timestamp_to_seq)

        print(f"Found {df['sequence_id'].nunique()} continuous sequences")

        return df

    def build_sequence_structure(self, df):
        """Build structured sequence data"""
        print("\nBuilding sequence structures...")

        sequences = []

        for seq_id in tqdm(sorted(df['sequence_id'].unique()), desc="Processing sequences"):
            seq_df = df[df['sequence_id'] == seq_id].copy()

            # Get unique timestamps in this sequence
            timestamps = sorted(seq_df['timestamp'].unique())

            # Calculate gap durations
            gap_durations = []
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0  # minutes
                gap_durations.append(gap)

            # Group by timestamp to get frames
            frames = []
            for ts in timestamps:
                frame_df = seq_df[seq_df['timestamp'] == ts]

                # Get all storms in this frame
                storms = []
                for idx, row in frame_df.iterrows():
                    storm = {
                        'bbox_id': int(row['bbox_id']),
                        'position': (float(row['x_center']), float(row['y_center'])),
                        'size': (float(row['bbox_width']), float(row['bbox_height'])),
                        'area': float(row['bbox_area']),
                        'aspect_ratio': float(row['aspect_ratio']),
                        'thermal': {
                            'mean_color': float(row['mean_color']),
                            'max_color': float(row['max_color']),
                            'min_color': float(row['min_color']),
                            'std_color': float(row['std_color']),
                            'color_gradient': float(row['color_gradient'])
                        }
                    }
                    storms.append(storm)

                frames.append({
                    'timestamp': ts,
                    'filename': frame_df.iloc[0]['filename'],
                    'storms': storms
                })

            # Create sequence object
            sequence = {
                'sequence_id': f"seq_{seq_id:03d}",
                'start_time': timestamps[0],
                'end_time': timestamps[-1],
                'duration': timestamps[-1] - timestamps[0],
                'num_frames': len(timestamps),
                'timestamps': timestamps,
                'gap_durations': gap_durations,
                'frames': frames,
                'stats': {
                    'mean_storms_per_frame': len(seq_df) / len(timestamps),
                    'total_storm_detections': len(seq_df),
                    'mean_gap_minutes': np.mean(gap_durations) if gap_durations else 0,
                    'max_gap_minutes': np.max(gap_durations) if gap_durations else 0
                }
            }

            sequences.append(sequence)

        print(f"Built {len(sequences)} sequences")

        return sequences

    def save_sequences(self, sequences, output_dir="data/processed/sequences"):
        """Save sequence data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as pickle (preserves datetime objects)
        pkl_path = output_dir / 'sequences.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(sequences, f)

        print(f"\n✅ Sequences saved to: {pkl_path}")
        print(f"   Size: {pkl_path.stat().st_size / 1024:.2f} KB")

        # Save metadata as JSON
        metadata = {
            'total_sequences': len(sequences),
            'sequences': []
        }

        for seq in sequences:
            metadata['sequences'].append({
                'sequence_id': seq['sequence_id'],
                'start_time': seq['start_time'].isoformat(),
                'end_time': seq['end_time'].isoformat(),
                'duration': str(seq['duration']),
                'num_frames': seq['num_frames'],
                'stats': seq['stats']
            })

        json_path = output_dir / 'sequences_metadata.json'
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   Metadata saved to: {json_path}")

        return pkl_path

    def print_summary(self, sequences):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SEQUENCE SUMMARY")
        print("="*80)

        total_frames = sum(s['num_frames'] for s in sequences)
        total_storms = sum(s['stats']['total_storm_detections'] for s in sequences)

        print(f"Total sequences: {len(sequences)}")
        print(f"Total frames: {total_frames}")
        print(f"Total storm detections: {total_storms}")
        print(f"Average storms per frame: {total_storms / total_frames:.2f}")

        print("\nTop 5 longest sequences:")
        sorted_seqs = sorted(sequences, key=lambda s: s['num_frames'], reverse=True)[:5]
        for i, seq in enumerate(sorted_seqs, 1):
            print(f"  {i}. {seq['sequence_id']}: {seq['num_frames']} frames, "
                  f"{seq['start_time']} to {seq['end_time']} "
                  f"(Duration: {seq['duration']})")


def main():
    builder = SequenceBuilder()

    # Load features
    df = builder.load_features()

    # Detect sequences
    df = builder.detect_sequences(df)

    # Build sequence structure
    sequences = builder.build_sequence_structure(df)

    # Print summary
    builder.print_summary(sequences)

    # Save sequences
    builder.save_sequences(sequences)

    print("\n✅ Sequence building complete!")


if __name__ == "__main__":
    main()
