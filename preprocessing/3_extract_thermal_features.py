"""
Extract thermal features from satellite images within bounding boxes.

Extracts pixel color values (grayscale or RGB) from regions defined by YOLO bboxes:
- mean_color, max_color, min_color
- std_color (variation - storm intensity proxy)
- color_gradient (center vs edges)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


class ThermalFeatureExtractor:
    def __init__(self,
                 combined_dir="data/raw/combined",
                 spatial_features_path="data/processed/features/spatial_features.parquet",
                 use_grayscale=True):
        self.combined_dir = Path(combined_dir)
        self.images_dir = self.combined_dir / 'images'
        self.spatial_features_path = Path(spatial_features_path)
        self.use_grayscale = use_grayscale

    def extract_bbox_region(self, image, x_center, y_center, width, height):
        """Extract image region within bounding box"""
        h, w = image.shape[:2]

        # Convert normalized coordinates to pixels
        x_center_px = int(x_center * w)
        y_center_px = int(y_center * h)
        width_px = int(width * w)
        height_px = int(height * h)

        # Calculate bbox corners
        x1 = max(0, x_center_px - width_px // 2)
        y1 = max(0, y_center_px - height_px // 2)
        x2 = min(w, x_center_px + width_px // 2)
        y2 = min(h, y_center_px + height_px // 2)

        # Extract region
        bbox_region = image[y1:y2, x1:x2]

        return bbox_region, (x1, y1, x2, y2)

    def compute_thermal_features(self, bbox_region):
        """Compute thermal statistics from bbox region"""
        if bbox_region.size == 0:
            return {
                'mean_color': np.nan,
                'max_color': np.nan,
                'min_color': np.nan,
                'std_color': np.nan,
                'color_gradient': np.nan
            }

        # Flatten to 1D array
        if len(bbox_region.shape) == 3:
            # RGB - convert to grayscale or use mean
            pixels = bbox_region.mean(axis=2).flatten()
        else:
            pixels = bbox_region.flatten()

        # Basic statistics
        mean_color = float(np.mean(pixels))
        max_color = float(np.max(pixels))
        min_color = float(np.min(pixels))
        std_color = float(np.std(pixels))

        # Gradient: center vs edges
        h, w = bbox_region.shape[:2]
        if h > 4 and w > 4:
            # Center region (middle 50%)
            center_h = slice(h//4, 3*h//4)
            center_w = slice(w//4, 3*w//4)

            if len(bbox_region.shape) == 3:
                center_region = bbox_region[center_h, center_w].mean(axis=2)
            else:
                center_region = bbox_region[center_h, center_w]

            center_mean = float(np.mean(center_region))

            # Edge region (outer ring)
            mask = np.ones((h, w), dtype=bool)
            mask[center_h, center_w] = False

            if len(bbox_region.shape) == 3:
                edge_pixels = bbox_region.mean(axis=2)[mask]
            else:
                edge_pixels = bbox_region[mask]

            edge_mean = float(np.mean(edge_pixels))

            # Gradient = difference between center and edges
            color_gradient = center_mean - edge_mean
        else:
            color_gradient = 0.0

        return {
            'mean_color': mean_color,
            'max_color': max_color,
            'min_color': min_color,
            'std_color': std_color,
            'color_gradient': color_gradient
        }

    def extract_all_features(self):
        """Extract thermal features for all bboxes"""
        print("="*80)
        print("EXTRACTING THERMAL FEATURES FROM IMAGES")
        print("="*80)

        # Load spatial features (contains bbox locations)
        print(f"\nLoading spatial features from: {self.spatial_features_path}")
        df_spatial = pd.read_parquet(self.spatial_features_path)
        print(f"Loaded {len(df_spatial)} bounding boxes")

        # Group by filename
        grouped = df_spatial.groupby('filename')

        thermal_features = []

        print("\nExtracting thermal features...")
        for filename, group in tqdm(grouped, desc="Processing images"):
            # Load image
            image_path = self.images_dir / f"{filename}.jpg"

            if not image_path.exists():
                print(f"  Warning: Image not found: {image_path}")
                continue

            # Read image
            if self.use_grayscale:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(str(image_path))

            if image is None:
                print(f"  Warning: Failed to load image: {image_path}")
                continue

            # Process each bbox in this image
            for idx, row in group.iterrows():
                # Extract bbox region
                bbox_region, coords = self.extract_bbox_region(
                    image,
                    row['x_center'],
                    row['y_center'],
                    row['bbox_width'],
                    row['bbox_height']
                )

                # Compute thermal features
                thermal = self.compute_thermal_features(bbox_region)

                # Combine with metadata
                feature_row = {
                    'filename': filename,
                    'timestamp': row['timestamp'],
                    'bbox_id': row['bbox_id'],
                    **thermal
                }

                thermal_features.append(feature_row)

        # Create DataFrame
        df_thermal = pd.DataFrame(thermal_features)

        print(f"\nExtracted thermal features for {len(df_thermal)} bounding boxes")

        # Display statistics
        print("\n" + "="*80)
        print("THERMAL FEATURE STATISTICS")
        print("="*80)
        print(df_thermal[['mean_color', 'max_color', 'min_color', 'std_color', 'color_gradient']].describe())

        return df_thermal

    def save_features(self, df, output_dir="data/processed/features"):
        """Save extracted thermal features"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'thermal_features.parquet'
        df.to_parquet(output_path, index=False)

        print(f"\n✅ Thermal features saved to: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")

        # Also save as CSV
        csv_path = output_dir / 'thermal_features.csv'
        df.to_csv(csv_path, index=False)
        print(f"   CSV copy saved to: {csv_path}")

        return output_path


def main():
    extractor = ThermalFeatureExtractor()

    # Extract features
    df = extractor.extract_all_features()

    # Save features
    extractor.save_features(df)

    print("\n✅ Thermal feature extraction complete!")


if __name__ == "__main__":
    main()
