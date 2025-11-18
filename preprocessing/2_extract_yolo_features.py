"""
Extract spatial features from YOLO bounding box annotations.

Extracts:
- x_center, y_center (normalized [0-1])
- bbox_width, bbox_height (normalized)
- bbox_area (derived)
- aspect_ratio (width/height)
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm


class YOLOFeatureExtractor:
    def __init__(self, combined_dir="data/raw/combined"):
        self.combined_dir = Path(combined_dir)
        self.labels_dir = self.combined_dir / 'labels'
        self.images_dir = self.combined_dir / 'images'

    def parse_timestamp_from_filename(self, filename):
        """
        Parse timestamp from filename format: YYYYDDDHHMM_...
        Example: 20232941410 -> Year 2023, Day 294, Hour 14, Minute 10
        """
        match = re.match(r'(\d{4})(\d{3})(\d{2})(\d{2})', filename)
        if match:
            year = int(match.group(1))
            day_of_year = int(match.group(2))
            hour = int(match.group(3))
            minute = int(match.group(4))

            # Convert day of year to datetime
            dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour, minutes=minute)
            return dt
        return None

    def extract_features_from_label(self, label_path):
        """Extract features from a single YOLO label file"""
        bboxes = []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)

                # Calculate derived features
                bbox_area = width * height
                aspect_ratio = width / height if height > 0 else 0

                bboxes.append({
                    'class_id': int(class_id),
                    'x_center': x_center,
                    'y_center': y_center,
                    'bbox_width': width,
                    'bbox_height': height,
                    'bbox_area': bbox_area,
                    'aspect_ratio': aspect_ratio
                })

        return bboxes

    def extract_all_features(self):
        """Extract features from all label files"""
        print("="*80)
        print("EXTRACTING YOLO SPATIAL FEATURES")
        print("="*80)

        label_files = sorted(self.labels_dir.glob('*.txt'))
        print(f"\nFound {len(label_files)} label files")

        all_data = []

        for label_path in tqdm(label_files, desc="Extracting features"):
            # Get filename and timestamp
            filename = label_path.stem
            timestamp = self.parse_timestamp_from_filename(filename)

            # Extract bboxes
            bboxes = self.extract_features_from_label(label_path)

            # Add to dataset
            for bbox_idx, bbox in enumerate(bboxes):
                row = {
                    'filename': filename,
                    'timestamp': timestamp,
                    'bbox_id': bbox_idx,
                    **bbox
                }
                all_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Sort by timestamp and bbox_id
        df = df.sort_values(['timestamp', 'bbox_id']).reset_index(drop=True)

        print(f"\nExtracted {len(df)} bounding boxes from {len(label_files)} files")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Average bboxes per image: {len(df) / len(label_files):.2f}")

        # Display statistics
        print("\n" + "="*80)
        print("SPATIAL FEATURE STATISTICS")
        print("="*80)
        print(df[['x_center', 'y_center', 'bbox_width', 'bbox_height', 'bbox_area', 'aspect_ratio']].describe())

        return df

    def save_features(self, df, output_dir="data/processed/features"):
        """Save extracted features to parquet file"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'spatial_features.parquet'
        df.to_parquet(output_path, index=False)

        print(f"\n✅ Spatial features saved to: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")

        # Also save as CSV for easy inspection
        csv_path = output_dir / 'spatial_features.csv'
        df.to_csv(csv_path, index=False)
        print(f"   CSV copy saved to: {csv_path}")

        return output_path


def main():
    extractor = YOLOFeatureExtractor()

    # Extract features
    df = extractor.extract_all_features()

    # Save features
    extractor.save_features(df)

    print("\n✅ YOLO feature extraction complete!")


if __name__ == "__main__":
    main()
