"""
Auto-Annotate Images with YOLO11

Uses trained YOLO model to automatically annotate new hurricane images.
"""

from ultralytics import YOLO
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np


class YOLOAutoAnnotator:
    """Automatic annotation using trained YOLO model"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5, verbose: bool = True):
        """
        Initialize auto-annotator.

        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence_threshold: Minimum detection confidence (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            verbose: Print progress messages
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose

        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'images_with_detections': 0,
            'images_without_detections': 0,
            'confidence_distribution': [],
            'detections_per_image': []
        }

    def annotate_image(self, image_path: Path) -> List[Dict]:
        """
        Annotate a single image.

        Args:
            image_path: Path to image file

        Returns:
            List of detections, each with class, bbox, and confidence
        """
        # Run inference
        results = self.model.predict(
            image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []

        # Extract bounding boxes
        for box in results[0].boxes:
            # Get normalized coordinates (YOLO format: x_center, y_center, width, height)
            xywhn = box.xywhn[0].cpu().numpy()
            x_center, y_center, width, height = xywhn

            detection = {
                'class': int(box.cls[0].cpu().numpy()),
                'bbox': {
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height)
                },
                'confidence': float(box.conf[0].cpu().numpy())
            }

            detections.append(detection)
            self.stats['confidence_distribution'].append(detection['confidence'])

        return detections

    def save_yolo_label(self, detections: List[Dict], output_path: Path):
        """
        Save detections in YOLO label format.

        Args:
            detections: List of detection dictionaries
            output_path: Path to save .txt label file
        """
        with open(output_path, 'w') as f:
            for det in detections:
                # YOLO format: class x_center y_center width height
                class_id = det['class']
                bbox = det['bbox']
                f.write(f"{class_id} {bbox['x_center']} {bbox['y_center']} "
                       f"{bbox['width']} {bbox['height']}\n")

    def annotate_directory(self, image_dir: Path, output_label_dir: Path,
                          save_metadata: bool = True) -> Dict:
        """
        Auto-annotate all images in a directory.

        Args:
            image_dir: Directory containing images
            output_label_dir: Directory to save YOLO label files
            save_metadata: Save annotation metadata JSON

        Returns:
            Dictionary with annotation statistics
        """
        image_dir = Path(image_dir)
        output_label_dir = Path(output_label_dir)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))

        if len(image_files) == 0:
            print(f"No images found in {image_dir}")
            return self.stats

        self.stats['total_images'] = len(image_files)

        if self.verbose:
            print("="*80)
            print("AUTO-ANNOTATION")
            print("="*80)
            print(f"Model: {self.model_path.name}")
            print(f"Confidence threshold: {self.confidence_threshold}")
            print(f"IoU threshold: {self.iou_threshold}")
            print(f"Images to process: {len(image_files)}")
            print(f"Output: {output_label_dir}")
            print("="*80)
            print()

        start_time = datetime.now()
        annotation_metadata = []

        # Process each image
        for i, image_path in enumerate(image_files):
            # Annotate image
            detections = self.annotate_image(image_path)

            # Save label file
            label_path = output_label_dir / f"{image_path.stem}.txt"
            self.save_yolo_label(detections, label_path)

            # Update stats
            num_detections = len(detections)
            self.stats['total_detections'] += num_detections
            self.stats['detections_per_image'].append(num_detections)

            if num_detections > 0:
                self.stats['images_with_detections'] += 1
            else:
                self.stats['images_without_detections'] += 1

            # Save metadata
            annotation_metadata.append({
                'image': image_path.name,
                'label_file': label_path.name,
                'num_detections': num_detections,
                'confidences': [d['confidence'] for d in detections],
                'mean_confidence': float(np.mean([d['confidence'] for d in detections])) if detections else 0.0
            })

            # Progress update
            if self.verbose and (i + 1) % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed
                remaining = (len(image_files) - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(image_files)} ({(i+1)/len(image_files)*100:.1f}%)")
                print(f"  Detections: {self.stats['total_detections']}, "
                      f"Images with storms: {self.stats['images_with_detections']}")
                print(f"  Rate: {rate:.1f} images/sec, ETA: {remaining/60:.1f} minutes\n")

        elapsed = (datetime.now() - start_time).total_seconds()

        # Calculate statistics
        if self.stats['confidence_distribution']:
            conf_array = np.array(self.stats['confidence_distribution'])
            self.stats['confidence_mean'] = float(np.mean(conf_array))
            self.stats['confidence_std'] = float(np.std(conf_array))
            self.stats['confidence_min'] = float(np.min(conf_array))
            self.stats['confidence_max'] = float(np.max(conf_array))

        if self.stats['detections_per_image']:
            det_array = np.array(self.stats['detections_per_image'])
            self.stats['detections_mean'] = float(np.mean(det_array))
            self.stats['detections_std'] = float(np.std(det_array))
            self.stats['detections_max'] = int(np.max(det_array))

        self.stats['elapsed_seconds'] = elapsed

        # Save metadata
        if save_metadata:
            metadata_path = output_label_dir.parent / 'annotation_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model': str(self.model_path),
                    'confidence_threshold': self.confidence_threshold,
                    'iou_threshold': self.iou_threshold,
                    'timestamp': datetime.now().isoformat(),
                    'statistics': self.stats,
                    'annotations': annotation_metadata
                }, f, indent=2)

            if self.verbose:
                print(f"Metadata saved: {metadata_path}")

        # Print summary
        if self.verbose:
            print("\n" + "="*80)
            print("ANNOTATION COMPLETE")
            print("="*80)
            print(f"Total images: {self.stats['total_images']}")
            print(f"  With detections: {self.stats['images_with_detections']} "
                  f"({self.stats['images_with_detections']/self.stats['total_images']*100:.1f}%)")
            print(f"  Without detections: {self.stats['images_without_detections']} "
                  f"({self.stats['images_without_detections']/self.stats['total_images']*100:.1f}%)")
            print(f"\nTotal detections: {self.stats['total_detections']}")
            print(f"  Mean per image: {self.stats.get('detections_mean', 0):.2f} ± {self.stats.get('detections_std', 0):.2f}")
            print(f"  Max per image: {self.stats.get('detections_max', 0)}")
            print(f"\nConfidence statistics:")
            print(f"  Mean: {self.stats.get('confidence_mean', 0):.3f} ± {self.stats.get('confidence_std', 0):.3f}")
            print(f"  Range: [{self.stats.get('confidence_min', 0):.3f}, {self.stats.get('confidence_max', 0):.3f}]")
            print(f"\nElapsed time: {elapsed/60:.1f} minutes ({rate:.1f} images/sec)")
            print("="*80)

        return self.stats

    def create_yolo_dataset(self, image_dir: Path, output_dir: Path,
                           dataset_name: str = "auto_annotated"):
        """
        Create a complete YOLO dataset structure with auto-annotations.

        Args:
            image_dir: Directory containing images
            output_dir: Base output directory
            dataset_name: Name for the dataset

        Returns:
            Path to dataset YAML file
        """
        output_dir = Path(output_dir) / dataset_name
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        import shutil
        image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))

        if self.verbose:
            print(f"Copying {len(image_files)} images to {images_dir}...")

        for img in image_files:
            shutil.copy(img, images_dir / img.name)

        # Auto-annotate
        self.annotate_directory(images_dir, labels_dir, save_metadata=True)

        # Create data.yaml
        yaml_path = output_dir / 'data.yaml'
        yaml_content = f"""# Auto-annotated YOLO dataset
# Generated: {datetime.now().isoformat()}

path: {output_dir.absolute()}
train: images
val: images  # Use same for validation (or create separate validation set)

nc: 1  # Number of classes
names:
  0: Hurricane  # Class 0: Hurricane

# Model used: {self.model_path.name}
# Confidence threshold: {self.confidence_threshold}
# Total images: {self.stats['total_images']}
# Total detections: {self.stats['total_detections']}
"""

        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        if self.verbose:
            print(f"\nDataset YAML created: {yaml_path}")

        return yaml_path


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Auto-annotate images with YOLO11')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')

    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images to annotate')

    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for labels (default: images/../labels)')

    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0, default 0.5)')

    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS (default 0.5)')

    parser.add_argument('--create-dataset', action='store_true',
                       help='Create complete YOLO dataset structure')

    parser.add_argument('--dataset-name', type=str, default='auto_annotated',
                       help='Name for created dataset (if --create-dataset)')

    args = parser.parse_args()

    # Initialize annotator
    annotator = YOLOAutoAnnotator(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        verbose=True
    )

    image_dir = Path(args.images)

    if args.create_dataset:
        # Create complete dataset
        output_base = Path(args.output) if args.output else image_dir.parent / 'datasets'
        annotator.create_yolo_dataset(
            image_dir=image_dir,
            output_dir=output_base,
            dataset_name=args.dataset_name
        )
    else:
        # Just annotate
        output_label_dir = Path(args.output) if args.output else image_dir.parent / 'labels'
        annotator.annotate_directory(
            image_dir=image_dir,
            output_label_dir=output_label_dir,
            save_metadata=True
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # Show example usage
        print("YOLO11 Auto-Annotation Tool")
        print("="*80)
        print()
        print("Usage:")
        print("  python 7_auto_annotate.py --model path/to/best.pt --images path/to/images")
        print()
        print("Options:")
        print("  --output DIR          Output directory for labels")
        print("  --conf 0.5            Confidence threshold (default 0.5)")
        print("  --iou 0.5             IoU threshold (default 0.5)")
        print("  --create-dataset      Create complete YOLO dataset structure")
        print("  --dataset-name NAME   Name for dataset (default: auto_annotated)")
        print()
        print("Examples:")
        print("  # Basic annotation")
        print("  python 7_auto_annotate.py --model yolo11n.pt --images data/raw/noaa_test_2024")
        print()
        print("  # Create full dataset")
        print("  python 7_auto_annotate.py --model yolo11n.pt --images data/raw/noaa_test_2024 \\")
        print("      --create-dataset --dataset-name hurricane_2024")
        print()
        print("  # Lower confidence for more detections")
        print("  python 7_auto_annotate.py --model yolo11n.pt --images imgs --conf 0.3")
        print()
        print("="*80)
    else:
        main()
