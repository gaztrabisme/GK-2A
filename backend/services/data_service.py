"""
Data service wrapper around visualization data loader
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from visualization.forecast_viz import VisualizationDataLoader


class DataService:
    """
    Service layer for accessing hurricane forecast data.
    Wraps the existing VisualizationDataLoader.
    """

    def __init__(self):
        """Initialize data loader"""
        print("Initializing DataService...")
        self.loader = VisualizationDataLoader()
        print(f"✓ Loaded {self.loader.get_total_frames()} frames")

    def get_total_frames(self) -> int:
        """Get total number of frames"""
        return self.loader.get_total_frames()

    def get_frame_data(self, frame_idx: int) -> dict:
        """
        Get complete frame data including storms, predictions, and ground truth.

        Args:
            frame_idx: Frame index (0 to total_frames-1)

        Returns:
            Dictionary with frame data
        """
        if frame_idx < 0 or frame_idx >= self.get_total_frames():
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.get_total_frames()-1}]")

        # Get raw frame data from loader
        frame_data = self.loader.get_frame(frame_idx)

        # Calculate errors
        errors = self._calculate_errors(frame_data)

        # Format response
        return {
            "frame_idx": frame_data["frame_idx"],
            "timestamp": str(frame_data["timestamp"]),
            "sequence_id": frame_data["sequence_id"],
            "filename": frame_data["filename"],
            "image_path": frame_data.get("image_path", ""),
            "storms": [
                {
                    "track_id": storm["track_id"],
                    "sequence_id": storm["sequence_id"],
                    "frame_idx": storm["frame_idx"],
                    "bbox": storm["bbox"]
                }
                for storm in frame_data["storms"]
            ],
            "predictions": self._format_predictions(frame_data["predictions"]),
            "ground_truth": self._format_ground_truth(frame_data["ground_truth_future"]),
            "errors": errors
        }

    def get_image_path(self, frame_idx: int) -> str:
        """
        Get path to satellite image for frame.

        Args:
            frame_idx: Frame index

        Returns:
            Absolute path to image file
        """
        frame_data = self.loader.get_frame(frame_idx)
        return frame_data.get("image_path", "")

    def get_metadata(self) -> dict:
        """
        Get dataset metadata.

        Returns:
            Dictionary with metadata
        """
        # Get date range from first and last frame
        first_frame = self.loader.get_frame(0)
        last_frame = self.loader.get_frame(self.get_total_frames() - 1)

        # Get unique sequences
        sequences = set()
        for frame in self.loader.frames:
            sequences.add(frame["sequence_id"])

        return {
            "total_frames": self.get_total_frames(),
            "sequences": sorted(list(sequences)),
            "date_range": {
                "start": str(first_frame["timestamp"]),
                "end": str(last_frame["timestamp"])
            },
            "horizons": ["t+1", "t+3", "t+6", "t+12"],
            "confidence_scores": {
                "t+1": 0.8623,
                "t+3": 0.8173,
                "t+6": 0.7611,
                "t+12": 0.5949
            }
        }

    def _format_predictions(self, predictions: dict) -> dict:
        """Format predictions for API response"""
        formatted = {}

        for track_id, pred_dict in predictions.items():
            formatted[track_id] = {"track_id": track_id}

            for horizon, pred in pred_dict.items():
                if "confidence" in pred:
                    formatted[track_id][horizon] = {
                        "x": float(pred["x"]),
                        "y": float(pred["y"]),
                        "confidence": float(pred["confidence"]),
                        "exists": True
                    }
                else:
                    formatted[track_id][horizon] = {
                        "exists": False
                    }

        return formatted

    def _format_ground_truth(self, ground_truth: dict) -> dict:
        """Format ground truth for API response"""
        formatted = {}

        for track_id, gt_dict in ground_truth.items():
            formatted[track_id] = {"track_id": track_id}

            for horizon, gt in gt_dict.items():
                if gt.get("exists", False):
                    formatted[track_id][horizon] = {
                        "x": float(gt["x"]),
                        "y": float(gt["y"]),
                        "exists": True
                    }
                else:
                    formatted[track_id][horizon] = {
                        "exists": False
                    }

        return formatted

    def _calculate_errors(self, frame_data: dict) -> dict:
        """Calculate error metrics for all storms and horizons"""
        errors = {}

        for storm in frame_data["storms"]:
            track_id = storm["track_id"]
            storm_predictions = frame_data["predictions"].get(track_id, {})
            storm_ground_truth = frame_data["ground_truth_future"].get(track_id, {})

            storm_errors = {"track_id": track_id}

            for horizon in ["t+1", "t+3", "t+6", "t+12"]:
                if horizon in storm_predictions and horizon in storm_ground_truth:
                    pred = storm_predictions[horizon]
                    actual = storm_ground_truth[horizon]

                    if pred.get("confidence") and actual.get("exists"):
                        # Calculate Euclidean distance
                        dx = pred["x"] - actual["x"]
                        dy = pred["y"] - actual["y"]
                        euclidean = np.sqrt(dx**2 + dy**2)

                        # Convert to pixels (assuming 678x678 image)
                        error_pixels = euclidean * 678

                        # As percentage of image size
                        error_pct = euclidean * 100

                        storm_errors[horizon] = {
                            "error_pct": float(error_pct),
                            "error_pixels": float(error_pixels),
                            "euclidean_distance": float(euclidean)
                        }

            # Only add if there are actual error measurements (not just track_id)
            if len(storm_errors) > 1:
                errors[track_id] = storm_errors

        return errors


# Singleton instance
_data_service_instance = None


def get_data_service() -> DataService:
    """Get singleton DataService instance"""
    global _data_service_instance

    if _data_service_instance is None:
        _data_service_instance = DataService()

    return _data_service_instance
