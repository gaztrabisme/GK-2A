"""
Storm Tracker - Production Implementation
Based on analysis in storm_tracking_report.md

Usage:
    from storm_tracker import StormTracker

    tracker = StormTracker(distance_threshold=100)
    for frame in frames:
        tracker.process_frame(frame)

    tracks = tracker.get_tracks(min_length=3)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json


class StormTracker:
    """
    Multi-object storm tracker using Hungarian algorithm

    Tracks storms across consecutive frames using optimal global assignment.
    Handles storm appearances, disappearances, and variable time gaps.

    Attributes:
        distance_threshold (int): Maximum matching distance in pixels (default: 100)
        image_size (int): Image dimension in pixels (default: 678)
        tracks (List[Dict]): List of storm tracks
    """

    def __init__(self, distance_threshold: int = 100, image_size: int = 678):
        """
        Initialize storm tracker

        Args:
            distance_threshold: Maximum distance (pixels) for valid match
            image_size: Image dimension in pixels
        """
        self.distance_threshold = distance_threshold
        self.image_size = image_size
        self.tracks = []
        self.next_track_id = 0
        self.current_frame_idx = -1

    def euclidean_distance(self, storm1: Dict, storm2: Dict) -> float:
        """
        Calculate Euclidean distance between two storms

        Args:
            storm1: First storm dict with 'x' and 'y' keys (normalized coords)
            storm2: Second storm dict with 'x' and 'y' keys (normalized coords)

        Returns:
            Distance in pixels
        """
        dx = storm1['x'] - storm2['x']
        dy = storm1['y'] - storm2['y']
        return np.sqrt(dx**2 + dy**2) * self.image_size

    def get_adaptive_threshold(self, time_gap: float) -> float:
        """
        Adjust distance threshold based on time gap between frames

        Args:
            time_gap: Time gap in minutes

        Returns:
            Adjusted distance threshold in pixels
        """
        if time_gap <= 10:
            return self.distance_threshold
        elif time_gap <= 20:
            return self.distance_threshold * 1.5
        elif time_gap <= 30:
            return self.distance_threshold * 2.0
        else:
            # Don't track across large gaps
            return float('inf')

    def process_frame(
        self,
        storms: List[Dict],
        frame_idx: Optional[int] = None,
        prev_storms: Optional[List[Dict]] = None,
        time_gap: float = 10.0
    ) -> None:
        """
        Process a single frame and update tracks

        Args:
            storms: List of storm detections, each with 'x', 'y', 'width', 'height'
            frame_idx: Frame index (auto-increments if None)
            prev_storms: Previous frame's storms (for distance calculation)
            time_gap: Time gap from previous frame in minutes
        """
        # Auto-increment frame index
        if frame_idx is None:
            self.current_frame_idx += 1
            frame_idx = self.current_frame_idx
        else:
            self.current_frame_idx = frame_idx

        # First frame: initialize tracks
        if prev_storms is None or len(self.tracks) == 0:
            for storm_idx, storm in enumerate(storms):
                self._create_track(storm, storm_idx, frame_idx)
            return

        # Get active tracks
        active_indices = [i for i, t in enumerate(self.tracks) if t['active']]
        n_tracks = len(active_indices)
        n_storms = len(storms)

        # No active tracks: start new ones
        if n_tracks == 0:
            for storm_idx, storm in enumerate(storms):
                self._create_track(storm, storm_idx, frame_idx)
            return

        # Build cost matrix (tracks × current storms)
        cost_matrix = np.full((n_tracks, n_storms), 10000.0)

        for i, track_idx in enumerate(active_indices):
            track = self.tracks[track_idx]
            last_storm_idx = track['storm_indices'][-1]

            # Get last storm from previous frame
            if last_storm_idx < len(prev_storms):
                last_storm = prev_storms[last_storm_idx]

                # Calculate distance to each current storm
                for j, curr_storm in enumerate(storms):
                    dist = self.euclidean_distance(last_storm, curr_storm)
                    cost_matrix[i, j] = dist

        # Solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Get adaptive threshold
        threshold = self.get_adaptive_threshold(time_gap)

        # Mark all active tracks as inactive (will reactivate if matched)
        for track_idx in active_indices:
            self.tracks[track_idx]['active'] = False

        # Update matched tracks
        matched_storms = set()
        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i, j]

            # Only match if within threshold
            if cost < threshold:
                track_idx = active_indices[i]
                storm = storms[j]

                # Update track
                self.tracks[track_idx]['storm_indices'].append(j)
                self.tracks[track_idx]['frame_indices'].append(frame_idx)
                self.tracks[track_idx]['active'] = True
                self.tracks[track_idx]['end_frame'] = frame_idx

                # Store position and size
                self.tracks[track_idx]['positions'].append((storm['x'], storm['y']))
                self.tracks[track_idx]['sizes'].append((storm['width'], storm['height']))

                matched_storms.add(j)

        # Create new tracks for unmatched storms (new appearances)
        for storm_idx, storm in enumerate(storms):
            if storm_idx not in matched_storms:
                self._create_track(storm, storm_idx, frame_idx)

    def _create_track(self, storm: Dict, storm_idx: int, frame_idx: int) -> None:
        """
        Create a new track

        Args:
            storm: Storm detection dict
            storm_idx: Storm index in current frame
            frame_idx: Current frame index
        """
        self.tracks.append({
            'id': self.next_track_id,
            'storm_indices': [storm_idx],
            'frame_indices': [frame_idx],
            'active': True,
            'start_frame': frame_idx,
            'end_frame': frame_idx,
            'positions': [(storm['x'], storm['y'])],
            'sizes': [(storm['width'], storm['height'])]
        })
        self.next_track_id += 1

    def get_tracks(self, min_length: int = 3, active_only: bool = False) -> List[Dict]:
        """
        Get tracks matching criteria

        Args:
            min_length: Minimum number of detections
            active_only: Only return currently active tracks

        Returns:
            List of track dictionaries
        """
        filtered = self.tracks

        if active_only:
            filtered = [t for t in filtered if t['active']]

        if min_length > 0:
            filtered = [t for t in filtered if len(t['frame_indices']) >= min_length]

        return filtered

    def compute_velocities(self, track: Dict) -> List[Tuple[float, float]]:
        """
        Compute velocity vectors for a track

        Args:
            track: Track dictionary

        Returns:
            List of (vx, vy) velocity vectors in pixels/frame
        """
        positions = track['positions']
        velocities = []

        for i in range(1, len(positions)):
            vx = (positions[i][0] - positions[i-1][0]) * self.image_size
            vy = (positions[i][1] - positions[i-1][1]) * self.image_size
            velocities.append((vx, vy))

        return velocities

    def export_for_time_series(self, min_length: int = 3) -> List[Dict]:
        """
        Export tracks in format suitable for time-series modeling

        Args:
            min_length: Minimum track length

        Returns:
            List of track data dictionaries with features
        """
        long_tracks = self.get_tracks(min_length=min_length)
        export_data = []

        for track in long_tracks:
            # Compute velocities
            velocities = self.compute_velocities(track)

            # Compute accelerations
            accelerations = []
            for i in range(1, len(velocities)):
                ax = velocities[i][0] - velocities[i-1][0]
                ay = velocities[i][1] - velocities[i-1][1]
                accelerations.append((ax, ay))

            # Extract areas
            areas = [w * h for w, h in track['sizes']]

            track_data = {
                'track_id': track['id'],
                'length': len(track['frame_indices']),
                'start_frame': track['start_frame'],
                'end_frame': track['end_frame'],
                'frames': track['frame_indices'],
                'positions': track['positions'],  # (x, y) normalized
                'sizes': track['sizes'],  # (width, height) normalized
                'areas': areas,  # width * height
                'velocities': velocities,  # (vx, vy) pixels/frame
                'accelerations': accelerations,  # (ax, ay) pixels/frame²
            }

            export_data.append(track_data)

        return export_data

    def save_tracks(self, filepath: str, min_length: int = 3) -> None:
        """
        Save tracks to JSON file

        Args:
            filepath: Output file path
            min_length: Minimum track length to save
        """
        tracks_data = self.export_for_time_series(min_length=min_length)

        # Convert tuples to lists for JSON serialization
        for track in tracks_data:
            track['positions'] = [list(p) for p in track['positions']]
            track['sizes'] = [list(s) for s in track['sizes']]
            track['velocities'] = [list(v) for v in track['velocities']]
            track['accelerations'] = [list(a) for a in track['accelerations']]

        with open(filepath, 'w') as f:
            json.dump(tracks_data, f, indent=2)

        print(f"Saved {len(tracks_data)} tracks to {filepath}")

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of tracks

        Returns:
            Dictionary with statistics
        """
        all_tracks = self.tracks
        long_tracks = self.get_tracks(min_length=3)

        track_lengths = [len(t['frame_indices']) for t in all_tracks]

        return {
            'total_tracks': len(all_tracks),
            'long_tracks': len(long_tracks),
            'active_tracks': len([t for t in all_tracks if t['active']]),
            'mean_track_length': np.mean(track_lengths) if track_lengths else 0,
            'median_track_length': np.median(track_lengths) if track_lengths else 0,
            'max_track_length': max(track_lengths) if track_lengths else 0,
            'frames_processed': self.current_frame_idx + 1
        }

    def print_summary(self) -> None:
        """Print summary statistics"""
        stats = self.get_summary_statistics()
        print("\n" + "="*60)
        print("STORM TRACKER SUMMARY")
        print("="*60)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Total tracks: {stats['total_tracks']}")
        print(f"Long tracks (≥3 frames): {stats['long_tracks']}")
        print(f"Currently active: {stats['active_tracks']}")
        print(f"Mean track length: {stats['mean_track_length']:.2f} frames")
        print(f"Median track length: {stats['median_track_length']:.1f} frames")
        print(f"Longest track: {stats['max_track_length']} frames")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Example: Load YOLO labels and track storms

    def load_yolo_labels(filepath: Path) -> List[Dict]:
        """Load YOLO format labels"""
        storms = []
        if not filepath.exists():
            return storms

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    storms.append({
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4]),
                    })
        return storms

    # Initialize tracker
    tracker = StormTracker(distance_threshold=100, image_size=678)

    # Load all label files
    labels_dir = Path("/Users/GaryT/Documents/Work/AI/Research/USAC/GK-2A/Hurricane.v3i.yolov8/train/labels")
    label_files = sorted(labels_dir.glob("*.txt"))

    print(f"Processing {len(label_files)} frames...")

    prev_storms = None
    for i, filepath in enumerate(label_files):
        storms = load_yolo_labels(filepath)

        # Process frame
        tracker.process_frame(
            storms=storms,
            frame_idx=i,
            prev_storms=prev_storms,
            time_gap=10.0  # Assume 10-minute intervals
        )

        prev_storms = storms

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(label_files)} frames...")

    # Print summary
    tracker.print_summary()

    # Export tracks
    output_path = "/Users/GaryT/Documents/Work/AI/Research/USAC/GK-2A/storm_tracks.json"
    tracker.save_tracks(output_path, min_length=3)

    # Show example track
    tracks = tracker.export_for_time_series(min_length=5)
    if tracks:
        print(f"\nExample track (ID {tracks[0]['track_id']}):")
        print(f"  Length: {tracks[0]['length']} frames")
        print(f"  Frames: {tracks[0]['start_frame']} to {tracks[0]['end_frame']}")
        print(f"  Mean velocity: ({np.mean([v[0] for v in tracks[0]['velocities']]):.2f}, "
              f"{np.mean([v[1] for v in tracks[0]['velocities']]):.2f}) px/frame")
