"""
Combine train, valid, and test datasets for better temporal continuity.

Based on analysis, the original splits break temporal sequences artificially.
This script merges all datasets and prepares them for proper time-based splitting.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

class DatasetCombiner:
    def __init__(self, base_dir="data/raw/Hurricane.v3i.yolov8", output_dir="data/raw/combined"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.splits = ['train', 'valid', 'test']

    def combine_datasets(self):
        """Combine train/valid/test into single dataset"""
        print("="*80)
        print("COMBINING TRAIN/VALID/TEST DATASETS")
        print("="*80)

        # Create output directories
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)

        stats = {
            'total_images': 0,
            'total_labels': 0,
            'splits': {}
        }

        # Process each split
        for split in self.splits:
            print(f"\nProcessing {split} split...")

            images_dir = self.base_dir / split / 'images'
            labels_dir = self.base_dir / split / 'labels'

            if not images_dir.exists():
                print(f"  Warning: {images_dir} not found, skipping...")
                continue

            # Get all images
            images = list(images_dir.glob('*.jpg'))
            labels = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []

            print(f"  Found {len(images)} images, {len(labels)} labels")

            # Copy files
            for img in tqdm(images, desc=f"  Copying {split} images"):
                dest = self.output_dir / 'images' / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)

            for lbl in tqdm(labels, desc=f"  Copying {split} labels"):
                dest = self.output_dir / 'labels' / lbl.name
                if not dest.exists():
                    shutil.copy2(lbl, dest)

            stats['splits'][split] = {
                'images': len(images),
                'labels': len(labels)
            }
            stats['total_images'] += len(images)
            stats['total_labels'] += len(labels)

        # Save metadata
        metadata = {
            'combined_date': datetime.now().isoformat(),
            'source_directory': str(self.base_dir),
            'output_directory': str(self.output_dir),
            'statistics': stats
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "="*80)
        print("COMBINATION COMPLETE")
        print("="*80)
        print(f"Total images: {stats['total_images']}")
        print(f"Total labels: {stats['total_labels']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata saved: {metadata_path}")

        return stats

    def verify_integrity(self):
        """Verify that images and labels are paired correctly"""
        print("\nVerifying data integrity...")

        images_dir = self.output_dir / 'images'
        labels_dir = self.output_dir / 'labels'

        images = {img.stem for img in images_dir.glob('*.jpg')}
        labels = {lbl.stem for lbl in labels_dir.glob('*.txt')}

        missing_labels = images - labels
        orphan_labels = labels - images

        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")
        print(f"Missing labels: {len(missing_labels)}")
        print(f"Orphan labels: {len(orphan_labels)}")

        if missing_labels:
            print(f"\nWarning: {len(missing_labels)} images have no labels")
            # Save list
            with open(self.output_dir / 'missing_labels.txt', 'w') as f:
                for stem in sorted(missing_labels):
                    f.write(f"{stem}\n")

        if orphan_labels:
            print(f"\nWarning: {len(orphan_labels)} labels have no images")
            with open(self.output_dir / 'orphan_labels.txt', 'w') as f:
                for stem in sorted(orphan_labels):
                    f.write(f"{stem}\n")

        return {
            'total_images': len(images),
            'total_labels': len(labels),
            'missing_labels': len(missing_labels),
            'orphan_labels': len(orphan_labels)
        }


def main():
    combiner = DatasetCombiner()

    # Combine datasets
    stats = combiner.combine_datasets()

    # Verify integrity
    integrity = combiner.verify_integrity()

    print("\n✅ Dataset combination complete!")
    print(f"Combined {stats['total_images']} images from train/valid/test splits")
    print(f"Ready for temporal analysis and re-splitting")


if __name__ == "__main__":
    main()
