"""
Temporal delta feature extraction from tracked storms.

Computes frame-to-frame changes (deltas) for:
- Size (bbox_area)
- Thermal properties (mean_color, std_color)
- Speed (from motion features)

Requires: sequences.pkl, tracked_storms.json, motion_features.parquet
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


class TemporalDeltaExtractor:
    def __init__(
        self,
        sequences_path="data/processed/sequences/sequences.pkl",
        tracks_path="data/processed/storm_tracking/tracked_storms.json",
        motion_path="data/processed/features/motion_features.parquet",
        output_path="data/processed/features/temporal_features.parquet"
    ):
        self.sequences_path = Path(sequences_path)
        self.tracks_path = Path(tracks_path)
        self.motion_path = Path(motion_path)
        self.output_path = Path(output_path)

    def load_data(self):
        """Load all required data sources"""
        print("Loading data sources...")

        # Load sequences (has thermal features)
        with open(self.sequences_path, 'rb') as f:
            sequences = pickle.load(f)
        print(f"  Loaded {len(sequences)} sequences")

        # Load tracked storms (has track assignments)
        with open(self.tracks_path, 'r') as f:
            tracks = json.load(f)
        print(f"  Loaded {len(tracks)} storm tracks")

        # Load motion features (has speed)
        motion_df = pd.read_parquet(self.motion_path)
        print(f"  Loaded {len(motion_df)} motion feature rows")

        return sequences, tracks, motion_df

    def build_thermal_lookup(self, sequences: List[Dict]) -> Dict:
        """
        Build lookup table: sequence_id -> frame_idx -> bbox_id -> thermal features

        Returns:
            Nested dict for quick thermal feature lookup
        """
        print("\nBuilding thermal feature lookup table...")

        thermal_lookup = {}

        for seq in sequences:
            seq_id = seq['sequence_id']
            thermal_lookup[seq_id] = {}

            for frame_idx, frame in enumerate(seq['frames']):
                thermal_lookup[seq_id][frame_idx] = {}

                for storm in frame['storms']:
                    bbox_id = storm['bbox_id']
                    thermal_lookup[seq_id][frame_idx][bbox_id] = storm['thermal']

        total_entries = sum(
            len(frames)
            for seq_frames in thermal_lookup.values()
            for frames in seq_frames.values()
        )
        print(f"  Built lookup for {total_entries} storm detections")

        return thermal_lookup

    def build_bbox_to_track_mapping(self, sequences: List[Dict], tracks: List[Dict]) -> Dict:
        """
        Map (sequence_id, frame_idx, position) -> track_id

        This is needed because sequences have bbox_id but tracks use positions.
        We'll match by finding the closest position in each frame.

        Returns:
            Dict mapping (seq_id, frame_idx, bbox_id) -> track_id
        """
        print("\nBuilding bbox-to-track mapping...")

        # First, build track position lookup
        track_positions = {}
        for track in tracks:
            seq_id = track['sequence_id']
            track_id = track['track_id']
            frames = track['frames']
            positions = track['positions']

            if seq_id not in track_positions:
                track_positions[seq_id] = {}

            for frame_idx, pos in zip(frames, positions):
                if frame_idx not in track_positions[seq_id]:
                    track_positions[seq_id][frame_idx] = []
                track_positions[seq_id][frame_idx].append({
                    'track_id': track_id,
                    'position': pos
                })

        # Now map bbox_id to track_id by matching positions
        bbox_to_track = {}

        for seq in sequences:
            seq_id = seq['sequence_id']

            for frame_idx, frame in enumerate(seq['frames']):
                for storm in frame['storms']:
                    bbox_id = storm['bbox_id']
                    storm_pos = storm['position']

                    # Find matching track by position
                    if seq_id in track_positions and frame_idx in track_positions[seq_id]:
                        # Find closest track
                        min_dist = float('inf')
                        matched_track_id = None

                        for track_info in track_positions[seq_id][frame_idx]:
                            track_pos = track_info['position']
                            dist = np.sqrt(
                                (storm_pos[0] - track_pos[0])**2 +
                                (storm_pos[1] - track_pos[1])**2
                            )

                            if dist < min_dist:
                                min_dist = dist
                                matched_track_id = track_info['track_id']

                        # Only match if very close (should be exact or near-exact)
                        if min_dist < 0.001:  # threshold for normalized coords
                            bbox_to_track[(seq_id, frame_idx, bbox_id)] = matched_track_id

        print(f"  Mapped {len(bbox_to_track)} bbox detections to tracks")

        return bbox_to_track

    def extract_temporal_deltas(
        self,
        tracks: List[Dict],
        thermal_lookup: Dict,
        bbox_to_track: Dict,
        motion_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract temporal delta features for all tracks.

        Returns:
            DataFrame with delta features per frame
        """
        print("\n" + "="*80)
        print("EXTRACTING TEMPORAL DELTA FEATURES")
        print("="*80)

        feature_rows = []

        for track in tracks:
            track_id = track['track_id']
            seq_id = track['sequence_id']
            frames = track['frames']
            areas = track['areas']

            # Get speed from motion features
            track_motion = motion_df[motion_df['track_id'] == track_id].sort_values('frame_idx')
            speeds = track_motion['speed'].values

            # Process each frame
            for i, frame_idx in enumerate(frames):
                # First frame: no deltas
                if i == 0:
                    row = {
                        'track_id': track_id,
                        'sequence_id': seq_id,
                        'frame_idx': frame_idx,
                        'delta_area': np.nan,
                        'delta_mean_color': np.nan,
                        'delta_std_color': np.nan,
                        'delta_speed': np.nan
                    }

                # Subsequent frames: compute deltas
                else:
                    # Area delta
                    delta_area = areas[i] - areas[i-1]

                    # Get thermal features for current and previous frame
                    # We need to find the bbox_id for this track at these frames
                    curr_thermal = None
                    prev_thermal = None

                    # Search for matching bbox_id
                    for (s_id, f_idx, b_id), t_id in bbox_to_track.items():
                        if t_id == track_id and s_id == seq_id:
                            if f_idx == frame_idx and f_idx in thermal_lookup.get(seq_id, {}):
                                curr_thermal = thermal_lookup[seq_id][f_idx].get(b_id)
                            if f_idx == frames[i-1] and f_idx in thermal_lookup.get(seq_id, {}):
                                prev_thermal = thermal_lookup[seq_id][f_idx].get(b_id)

                    # Thermal deltas
                    if curr_thermal and prev_thermal:
                        delta_mean_color = curr_thermal['mean_color'] - prev_thermal['mean_color']
                        delta_std_color = curr_thermal['std_color'] - prev_thermal['std_color']
                    else:
                        delta_mean_color = np.nan
                        delta_std_color = np.nan

                    # Speed delta
                    if i < len(speeds) and i-1 < len(speeds):
                        delta_speed = speeds[i] - speeds[i-1]
                    else:
                        delta_speed = np.nan

                    row = {
                        'track_id': track_id,
                        'sequence_id': seq_id,
                        'frame_idx': frame_idx,
                        'delta_area': delta_area,
                        'delta_mean_color': delta_mean_color,
                        'delta_std_color': delta_std_color,
                        'delta_speed': delta_speed
                    }

                feature_rows.append(row)

        df = pd.DataFrame(feature_rows)

        print(f"\nExtracted temporal deltas for {len(feature_rows)} frame-storm pairs")
        print(f"Tracks: {df['track_id'].nunique()}")
        print(f"Sequences: {df['sequence_id'].nunique()}")

        return df

    def print_statistics(self, df: pd.DataFrame):
        """Print delta feature statistics"""
        print("\n" + "="*80)
        print("TEMPORAL DELTA STATISTICS")
        print("="*80)

        delta_cols = ['delta_area', 'delta_mean_color', 'delta_std_color', 'delta_speed']

        for col in delta_cols:
            valid_count = df[col].notna().sum()
            if valid_count > 0:
                print(f"\n{col}:")
                print(f"  Valid samples: {valid_count}")
                print(f"  Mean: {df[col].mean():.6f}")
                print(f"  Std: {df[col].std():.6f}")
                print(f"  Min: {df[col].min():.6f}")
                print(f"  Max: {df[col].max():.6f}")

                # Show distribution of positive/negative changes
                positive = (df[col] > 0).sum()
                negative = (df[col] < 0).sum()
                zero = (df[col] == 0).sum()
                print(f"  Positive changes: {positive}")
                print(f"  Negative changes: {negative}")
                print(f"  No change: {zero}")

    def save_features(self, df: pd.DataFrame):
        """Save temporal delta features to parquet"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)

        print("\n" + "="*80)
        print("SAVING TEMPORAL DELTA FEATURES")
        print("="*80)
        print(f"✅ Saved to: {self.output_path}")
        print(f"   Size: {self.output_path.stat().st_size / 1024:.2f} KB")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")

    def run(self):
        """Run complete temporal delta extraction pipeline"""
        # Load data
        sequences, tracks, motion_df = self.load_data()

        # Build lookup tables
        thermal_lookup = self.build_thermal_lookup(sequences)
        bbox_to_track = self.build_bbox_to_track_mapping(sequences, tracks)

        # Extract features
        df = self.extract_temporal_deltas(tracks, thermal_lookup, bbox_to_track, motion_df)

        # Print statistics
        self.print_statistics(df)

        # Save
        self.save_features(df)

        print("\n✅ Temporal delta feature extraction complete!")

        return df


def main():
    extractor = TemporalDeltaExtractor()
    df = extractor.run()

    print("\nTemporal delta features ready for model training!")
    print(f"Feature columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
