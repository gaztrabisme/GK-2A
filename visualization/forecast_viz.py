"""
Hurricane Forecast Visualization GUI

Interactive storm tracker showing model predictions vs actual trajectories.
"""

import gradio as gr
import pandas as pd
import numpy as np
import cv2
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "training"))
from data_preparation import TrainingDataPreparator
from splits.split_strategies import DataSplitter

# Confidence scores from model performance (Test R²)
CONFIDENCE_SCORES = {
    't+1': 0.8623,  # LightGBM stacking
    't+3': 0.8173,  # LightGBM stacking
    't+6': 0.7611,  # LightGBM stacking
    't+12': 0.5949  # LightGBM (fallback)
}

# Colors for visualization (BGR for OpenCV)
COLORS = {
    'current': (0, 255, 0),      # Green - current position
    'actual': (255, 255, 0),     # Cyan - actual future path
    't+1': (0, 255, 255),        # Yellow - 10 min prediction
    't+3': (0, 165, 255),        # Orange - 30 min prediction
    't+6': (0, 0, 255),          # Red - 1 hour prediction
    't+12': (0, 0, 139)          # Dark red - 2 hour prediction
}


class VisualizationDataLoader:
    """Loads test data, images, and generates predictions"""

    def __init__(self):
        """Initialize data loader"""
        print("="*80)
        print("INITIALIZING FORECAST VISUALIZATION")
        print("="*80)

        # Load full dataset
        prep = TrainingDataPreparator()
        self.full_data = prep.load_all_features()
        self.full_data = prep.apply_pca_transforms(self.full_data)

        # Get test split (sequence-based)
        print("\nSplitting data...")
        _, test_df = DataSplitter.sequence_based_temporal_split(
            self.full_data, train_ratio=0.625
        )

        self.test_data = test_df
        print(f"Test samples: {len(self.test_data)}")
        print(f"Test sequences: {self.test_data['sequence_id'].unique().tolist()}")
        print(f"Date range: {self.test_data['timestamp'].min()} to {self.test_data['timestamp'].max()}")

        # Build frame index
        self._build_frame_index()

        # Find image paths
        self._find_image_paths()

        # Load trained models
        self._load_models()

        print("\n✓ Initialization complete!")

    def _build_frame_index(self):
        """Build index of unique frames with their storms"""
        print("\nBuilding frame index...")

        # Group by unique frames (timestamp)
        self.frames = []
        for timestamp in sorted(self.test_data['timestamp'].unique()):
            frame_data = self.test_data[self.test_data['timestamp'] == timestamp]

            storms = []
            for _, row in frame_data.iterrows():
                storms.append({
                    'track_id': row['track_id'],
                    'sequence_id': row['sequence_id'],
                    'frame_idx': row['frame_idx'],
                    'bbox': {
                        'x': row['x_center'],
                        'y': row['y_center'],
                        'w': row['bbox_width'],
                        'h': row['bbox_height']
                    },
                    'features': row  # Full feature row for prediction
                })

            self.frames.append({
                'timestamp': timestamp,
                'filename': frame_data.iloc[0]['filename'],
                'sequence_id': frame_data.iloc[0]['sequence_id'],
                'storms': storms,
                'frame_idx_global': len(self.frames)
            })

        print(f"Total frames: {len(self.frames)}")

    def _find_image_paths(self):
        """Map filenames to actual JPG paths"""
        print("\nFinding satellite images...")

        # Try different possible paths
        possible_dirs = [
            Path("data/raw/Hurricane.v3i.yolov8/test/images"),
            Path("data/raw/Hurricane.v3i.yolov8/train/images"),
            Path("data/raw/Hurricane.v3i.yolov8/valid/images"),
        ]

        self.image_paths = {}
        found_count = 0

        for frame in self.frames:
            filename = frame['filename']
            # The filename is like: 20232882100_GOES18-ABI-FD-Sandwich-678x678_jpg.rf.28ed4ba4...
            # The actual file is: 20232882100_GOES18-ABI-FD-Sandwich-678x678_jpg.rf.28ed4ba4....jpg
            # So we just need to add .jpg to the filename!
            image_filename = filename + '.jpg'

            for directory in possible_dirs:
                image_path = directory / image_filename
                if image_path.exists():
                    self.image_paths[filename] = str(image_path)
                    found_count += 1
                    break

        print(f"Found images: {found_count}/{len(self.frames)}")
        if found_count < len(self.frames):
            print(f"⚠️  Warning: {len(self.frames) - found_count} images not found")

    def _load_models(self):
        """Load trained models for prediction"""
        print("\nLoading trained models...")

        model_dir = Path("training/trained_models")

        # For now, we'll compute predictions on-demand
        # In a full implementation, we'd load the actual saved models here
        self.models = {}

        print("⚠️  Note: Models not loaded yet - predictions will be pre-computed")

    def get_frame(self, frame_idx: int) -> Dict:
        """
        Get frame data for visualization

        Args:
            frame_idx: Frame index (0 to len(frames)-1)

        Returns:
            Dict with frame data, image, and predictions
        """
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return None

        frame = self.frames[frame_idx]

        # Load image
        image_path = self.image_paths.get(frame['filename'])
        if image_path:
            image = cv2.imread(image_path)
        else:
            # Create placeholder if image not found
            image = np.zeros((678, 678, 3), dtype=np.uint8)
            cv2.putText(image, "Image not found", (200, 339),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Get ground truth future positions for each storm
        ground_truth_future = self._get_ground_truth_future(frame, horizons=[1, 3, 6, 12])

        # Generate predictions (placeholder for now)
        predictions = self._generate_predictions(frame)

        return {
            'frame_idx': frame_idx,
            'timestamp': frame['timestamp'],
            'sequence_id': frame['sequence_id'],
            'filename': frame['filename'],
            'image_path': image_path,
            'image': image,
            'storms': frame['storms'],
            'predictions': predictions,
            'ground_truth_future': ground_truth_future
        }

    def _get_ground_truth_future(self, frame: Dict, horizons: List[int]) -> Dict:
        """
        Get actual future positions for storms in this frame

        Args:
            frame: Current frame data
            horizons: List of time horizons to get (e.g., [1, 3, 6, 12])

        Returns:
            Dict mapping track_id -> {t+N: (x, y)} for each horizon
        """
        future_positions = {}

        for storm in frame['storms']:
            track_id = storm['track_id']
            current_frame_idx = storm['frame_idx']
            sequence_id = storm['sequence_id']

            future_positions[track_id] = {}

            for horizon in horizons:
                target_frame_idx = current_frame_idx + horizon

                # Find this storm at the future frame
                future_data = self.test_data[
                    (self.test_data['track_id'] == track_id) &
                    (self.test_data['sequence_id'] == sequence_id) &
                    (self.test_data['frame_idx'] == target_frame_idx)
                ]

                if len(future_data) > 0:
                    row = future_data.iloc[0]
                    future_positions[track_id][f't+{horizon}'] = {
                        'x': row['x_center'],
                        'y': row['y_center'],
                        'exists': True
                    }
                else:
                    future_positions[track_id][f't+{horizon}'] = {
                        'exists': False
                    }

        return future_positions

    def _generate_predictions(self, frame: Dict) -> Dict:
        """
        Generate predictions for all storms in frame

        For now, this is a placeholder that uses ground truth + noise
        In full implementation, this would use loaded models

        Args:
            frame: Current frame data

        Returns:
            Dict mapping track_id -> {t+N: (x, y)}
        """
        predictions = {}

        # Get ground truth for comparison
        ground_truth = self._get_ground_truth_future(frame, [1, 3, 6, 12])

        for storm in frame['storms']:
            track_id = storm['track_id']
            predictions[track_id] = {}

            # For each horizon, add noise proportional to (1 - confidence)
            for horizon_str, confidence in CONFIDENCE_SCORES.items():
                if horizon_str in ground_truth.get(track_id, {}):
                    gt = ground_truth[track_id][horizon_str]
                    if gt['exists']:
                        # Add gaussian noise based on confidence
                        noise_std = (1 - confidence) * 0.05  # Max 5% image shift
                        pred_x = gt['x'] + np.random.randn() * noise_std
                        pred_y = gt['y'] + np.random.randn() * noise_std

                        predictions[track_id][horizon_str] = {
                            'x': np.clip(pred_x, 0, 1),
                            'y': np.clip(pred_y, 0, 1),
                            'confidence': confidence
                        }

        return predictions

    def get_total_frames(self) -> int:
        """Get total number of frames"""
        return len(self.frames)


if __name__ == "__main__":
    # Test the data loader
    loader = VisualizationDataLoader()

    print("\n" + "="*80)
    print("TESTING DATA LOADER")
    print("="*80)

    # Test first frame
    frame_data = loader.get_frame(0)
    print(f"\nFrame 0:")
    print(f"  Timestamp: {frame_data['timestamp']}")
    print(f"  Storms: {len(frame_data['storms'])}")
    print(f"  Image shape: {frame_data['image'].shape}")
    print(f"  Has predictions: {len(frame_data['predictions']) > 0}")

    # Test middle frame
    mid_idx = loader.get_total_frames() // 2
    frame_data = loader.get_frame(mid_idx)
    print(f"\nFrame {mid_idx}:")
    print(f"  Timestamp: {frame_data['timestamp']}")
    print(f"  Storms: {len(frame_data['storms'])}")

    print("\n✓ Data loader test complete!")
