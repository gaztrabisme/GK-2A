"""
Motion feature extraction from tracked storms.

Computes velocity, speed, direction, and acceleration features.
Requires storm tracking data (preprocessing step 5).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class MotionFeatureExtractor:
    def __init__(
        self,
        tracks_path="data/processed/storm_tracking/tracked_storms.json",
        output_path="data/processed/features/motion_features.parquet"
    ):
        self.tracks_path = Path(tracks_path)
        self.output_path = Path(output_path)

    def load_tracks(self) -> List[Dict]:
        """Load tracked storm data"""
        print("Loading tracked storms...")
        with open(self.tracks_path, 'r') as f:
            tracks = json.load(f)
        print(f"Loaded {len(tracks)} storm tracks")
        return tracks

    def compute_speed_direction(self, vx: float, vy: float) -> Tuple[float, float]:
        """
        Compute speed (magnitude) and direction (angle) from velocity components.

        Args:
            vx: Velocity in x direction (pixels/frame)
            vy: Velocity in y direction (pixels/frame)

        Returns:
            Tuple of (speed, direction_degrees)
        """
        speed = np.sqrt(vx**2 + vy**2)
        direction = np.degrees(np.arctan2(vy, vx))  # -180 to 180
        return speed, direction

    def compute_acceleration_magnitude(self, ax: float, ay: float) -> float:
        """
        Compute acceleration magnitude.

        Args:
            ax: Acceleration in x direction
            ay: Acceleration in y direction

        Returns:
            Magnitude of acceleration
        """
        return np.sqrt(ax**2 + ay**2)

    def extract_motion_features(self, tracks: List[Dict]) -> pd.DataFrame:
        """
        Extract motion features from all tracks.

        Creates one row per frame per storm with motion features.
        For frames where velocity/acceleration is not available (first/second frame),
        fills with NaN.

        Returns:
            DataFrame with columns:
                - track_id: Storm track identifier
                - sequence_id: Sequence this track belongs to
                - frame_idx: Frame index within sequence
                - velocity_x: X component of velocity (pixels/frame)
                - velocity_y: Y component of velocity (pixels/frame)
                - speed: Magnitude of velocity (pixels/frame)
                - direction: Angle of movement (degrees, -180 to 180)
                - acceleration_x: X component of acceleration
                - acceleration_y: Y component of acceleration
                - acceleration: Magnitude of acceleration
        """
        print("\n" + "="*80)
        print("EXTRACTING MOTION FEATURES")
        print("="*80)

        feature_rows = []

        for track in tracks:
            track_id = track['track_id']
            sequence_id = track['sequence_id']
            frames = track['frames']
            velocities = track['velocities']
            accelerations = track['accelerations']

            # Process each frame in the track
            for i, frame_idx in enumerate(frames):
                # Frame 0: No velocity/acceleration
                if i == 0:
                    row = {
                        'track_id': track_id,
                        'sequence_id': sequence_id,
                        'frame_idx': frame_idx,
                        'velocity_x': np.nan,
                        'velocity_y': np.nan,
                        'speed': np.nan,
                        'direction': np.nan,
                        'acceleration_x': np.nan,
                        'acceleration_y': np.nan,
                        'acceleration': np.nan
                    }

                # Frame 1: Velocity available, no acceleration
                elif i == 1:
                    vx, vy = velocities[i-1]
                    speed, direction = self.compute_speed_direction(vx, vy)

                    row = {
                        'track_id': track_id,
                        'sequence_id': sequence_id,
                        'frame_idx': frame_idx,
                        'velocity_x': vx,
                        'velocity_y': vy,
                        'speed': speed,
                        'direction': direction,
                        'acceleration_x': np.nan,
                        'acceleration_y': np.nan,
                        'acceleration': np.nan
                    }

                # Frame 2+: Both velocity and acceleration available
                else:
                    vx, vy = velocities[i-1]
                    ax, ay = accelerations[i-2]

                    speed, direction = self.compute_speed_direction(vx, vy)
                    accel_mag = self.compute_acceleration_magnitude(ax, ay)

                    row = {
                        'track_id': track_id,
                        'sequence_id': sequence_id,
                        'frame_idx': frame_idx,
                        'velocity_x': vx,
                        'velocity_y': vy,
                        'speed': speed,
                        'direction': direction,
                        'acceleration_x': ax,
                        'acceleration_y': ay,
                        'acceleration': accel_mag
                    }

                feature_rows.append(row)

        df = pd.DataFrame(feature_rows)

        print(f"\nExtracted motion features for {len(feature_rows)} frame-storm pairs")
        print(f"Tracks: {df['track_id'].nunique()}")
        print(f"Sequences: {df['sequence_id'].nunique()}")

        # Print statistics
        print("\n" + "="*80)
        print("MOTION FEATURE STATISTICS")
        print("="*80)

        stats_cols = ['velocity_x', 'velocity_y', 'speed', 'direction',
                      'acceleration_x', 'acceleration_y', 'acceleration']

        for col in stats_cols:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                if valid_count > 0:
                    print(f"\n{col}:")
                    print(f"  Valid samples: {valid_count}")
                    print(f"  Mean: {df[col].mean():.4f}")
                    print(f"  Std: {df[col].std():.4f}")
                    print(f"  Min: {df[col].min():.4f}")
                    print(f"  Max: {df[col].max():.4f}")

        return df

    def save_features(self, df: pd.DataFrame):
        """Save motion features to parquet"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)

        print("\n" + "="*80)
        print("SAVING MOTION FEATURES")
        print("="*80)
        print(f"✅ Saved to: {self.output_path}")
        print(f"   Size: {self.output_path.stat().st_size / 1024:.2f} KB")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")

    def run(self):
        """Run complete motion feature extraction pipeline"""
        # Load tracks
        tracks = self.load_tracks()

        # Extract features
        df = self.extract_motion_features(tracks)

        # Save
        self.save_features(df)

        print("\n✅ Motion feature extraction complete!")

        return df


def main():
    extractor = MotionFeatureExtractor()
    df = extractor.run()

    print("\nMotion features ready for model training!")
    print(f"Feature columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
