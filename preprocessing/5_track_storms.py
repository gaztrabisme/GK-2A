"""
Track storms across frames using Hungarian algorithm.

Integrates storm_tracker.py to assign unique IDs to storms across temporal sequences.
Outputs tracked storm data for time-series modeling.
"""

import pickle
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from storm_tracker import StormTracker


class StormTrackingPipeline:
    def __init__(self, sequences_path="data/processed/sequences/sequences.pkl"):
        self.sequences_path = Path(sequences_path)

    def load_sequences(self):
        """Load preprocessed sequences"""
        print("Loading sequences...")
        with open(self.sequences_path, 'rb') as f:
            sequences = pickle.load(f)
        print(f"Loaded {len(sequences)} sequences")
        return sequences

    def track_sequence(self, sequence, distance_threshold=100):
        """Track storms within a single sequence"""
        tracker = StormTracker(distance_threshold=distance_threshold, image_size=678)

        frames = sequence['frames']
        timestamps = sequence['timestamps']

        print(f"\n  Processing sequence {sequence['sequence_id']}")
        print(f"  Frames: {len(frames)}, Duration: {sequence['duration']}")

        # Process each frame
        for i, frame in enumerate(frames):
            # Prepare storms list for tracker
            curr_storms = []
            for storm in frame['storms']:
                tracker_storm = {
                    'x': storm['position'][0],
                    'y': storm['position'][1],
                    'width': storm['size'][0],
                    'height': storm['size'][1],
                    'area': storm['area'],
                    'thermal': storm['thermal']
                }
                curr_storms.append(tracker_storm)

            # Get previous frame storms
            prev_storms = None
            time_gap = 10  # default 10 minutes

            if i > 0:
                prev_storms = []
                for storm in frames[i-1]['storms']:
                    prev_storm = {
                        'x': storm['position'][0],
                        'y': storm['position'][1],
                        'width': storm['size'][0],
                        'height': storm['size'][1]
                    }
                    prev_storms.append(prev_storm)

                # Calculate time gap
                time_gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0

            # Process frame
            tracker.process_frame(curr_storms, i, prev_storms, time_gap)

        # Export tracks
        tracks = tracker.export_for_time_series(min_length=3)

        print(f"  Found {len(tracks)} tracks (≥3 frames)")
        if tracks:
            max_track = max(tracks, key=lambda t: t['length'])
            print(f"  Longest track: {max_track['length']} frames")

        return tracks

    def track_all_sequences(self, distance_threshold=100):
        """Track storms across all sequences"""
        print("="*80)
        print("STORM TRACKING PIPELINE")
        print("="*80)

        sequences = self.load_sequences()

        all_tracks = []
        track_stats = {
            'total_sequences': len(sequences),
            'total_tracks': 0,
            'tracks_per_sequence': [],
            'longest_track': 0,
            'mean_track_length': 0
        }

        for seq in sequences:
            tracks = self.track_sequence(seq, distance_threshold)

            # Add sequence context to tracks
            for track in tracks:
                track['sequence_id'] = seq['sequence_id']
                track['sequence_start'] = seq['start_time'].isoformat()
                track['sequence_end'] = seq['end_time'].isoformat()

            all_tracks.extend(tracks)
            track_stats['tracks_per_sequence'].append(len(tracks))

        # Calculate statistics
        track_stats['total_tracks'] = len(all_tracks)
        if all_tracks:
            track_lengths = [t['length'] for t in all_tracks]
            track_stats['longest_track'] = max(track_lengths)
            track_stats['mean_track_length'] = sum(track_lengths) / len(track_lengths)
            track_stats['min_track_length'] = min(track_lengths)

        print("\n" + "="*80)
        print("TRACKING SUMMARY")
        print("="*80)
        print(f"Total sequences processed: {track_stats['total_sequences']}")
        print(f"Total storm tracks found: {track_stats['total_tracks']}")
        print(f"Longest track: {track_stats['longest_track']} frames")
        print(f"Mean track length: {track_stats['mean_track_length']:.2f} frames")
        print(f"Min track length: {track_stats['min_track_length']} frames")

        return all_tracks, track_stats

    def save_tracks(self, tracks, stats, output_dir="data/processed/storm_tracking"):
        """Save tracked storm data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tracks as JSON
        tracks_path = output_dir / 'tracked_storms.json'
        with open(tracks_path, 'w') as f:
            json.dump(tracks, f, indent=2)

        print(f"\n✅ Tracked storms saved to: {tracks_path}")
        print(f"   Size: {tracks_path.stat().st_size / 1024:.2f} KB")

        # Save statistics
        stats_path = output_dir / 'tracking_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"   Statistics saved to: {stats_path}")

        # Print track distribution
        print("\n" + "="*80)
        print("TRACK LENGTH DISTRIBUTION")
        print("="*80)

        track_lengths = [t['length'] for t in tracks]
        length_bins = [
            (3, 5, "3-5 frames"),
            (6, 10, "6-10 frames"),
            (11, 20, "11-20 frames"),
            (21, 50, "21-50 frames"),
            (51, 100, "51-100 frames"),
            (101, float('inf'), "100+ frames")
        ]

        for min_len, max_len, label in length_bins:
            count = sum(1 for l in track_lengths if min_len <= l <= max_len)
            if count > 0:
                print(f"{label}: {count} tracks")

        return tracks_path


def main():
    pipeline = StormTrackingPipeline()

    # Track storms
    tracks, stats = pipeline.track_all_sequences(distance_threshold=100)

    # Save results
    pipeline.save_tracks(tracks, stats)

    print("\n✅ Storm tracking complete!")
    print(f"Total {stats['total_tracks']} storm tracks ready for time-series modeling")


if __name__ == "__main__":
    main()
