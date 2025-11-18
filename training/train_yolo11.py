"""
Train YOLO11 Hurricane Detector

Trains YOLO11 on the existing annotated dataset for auto-annotation of new images.
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import argparse
from datetime import datetime


class YOLOHurricaneTrainer:
    """Train YOLO11 model for hurricane detection"""

    def __init__(self, data_yaml: str = "data/raw/Hurricane.v3i.yolov8/data.yaml",
                 model_size: str = "n", output_dir: str = "training/yolo_models"):
        """
        Initialize trainer.

        Args:
            data_yaml: Path to YOLO dataset configuration
            model_size: YOLO11 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            output_dir: Directory to save trained models
        """
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model selection
        self.model_weights = {
            'n': 'yolo11n.pt',     # Nano: Fast, ~2-3M params, 80-85% mAP
            's': 'yolo11s.pt',     # Small: Balanced, ~9M params, 85-90% mAP
            'm': 'yolo11m.pt',     # Medium: Accurate, ~20M params, 90-95% mAP
            'l': 'yolo11l.pt',     # Large: Very accurate, ~25M params, 92-96% mAP
            'x': 'yolo11x.pt'      # XLarge: Best, ~68M params, 94-97% mAP
        }

        if model_size not in self.model_weights:
            raise ValueError(f"Invalid model size: {model_size}. "
                           f"Choose from {list(self.model_weights.keys())}")

        self.pretrained_weights = self.model_weights[model_size]

    def verify_dataset(self) -> dict:
        """
        Verify dataset exists and is properly configured.

        Returns:
            Dictionary with dataset statistics
        """
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.data_yaml}")

        # Load dataset config
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)

        # Verify paths exist
        dataset_root = self.data_yaml.parent
        train_path = dataset_root / config.get('train', 'train/images')
        val_path = dataset_root / config.get('val', 'valid/images')

        if not train_path.exists():
            raise FileNotFoundError(f"Training images not found: {train_path}")
        if not val_path.exists():
            print(f"Warning: Validation images not found: {val_path}")

        # Count images
        train_images = list(train_path.glob('*.jpg'))
        val_images = list(val_path.glob('*.jpg')) if val_path.exists() else []

        stats = {
            'train_images': len(train_images),
            'val_images': len(val_images),
            'classes': config.get('names', []),
            'num_classes': config.get('nc', 0)
        }

        print("="*80)
        print("DATASET VERIFICATION")
        print("="*80)
        print(f"Dataset config: {self.data_yaml}")
        print(f"Train images: {stats['train_images']}")
        print(f"Val images: {stats['val_images']}")
        print(f"Classes: {stats['classes']}")
        print(f"Number of classes: {stats['num_classes']}")
        print("="*80)
        print()

        return stats

    def train(self, epochs: int = 100, imgsz: int = 678, batch: int = 16,
             device: str = 'mps', patience: int = 20, save_period: int = 10,
             augment: bool = True, verbose: bool = True) -> dict:
        """
        Train YOLO11 model.

        Args:
            epochs: Number of training epochs
            imgsz: Image size (should match dataset, default 678)
            batch: Batch size (adjust based on GPU memory)
            device: 'mps' for Mac M-series, 'cuda' for NVIDIA, 'cpu' for CPU-only
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            augment: Use data augmentation
            verbose: Print detailed training info

        Returns:
            Dictionary with training results and metrics
        """
        print("="*80)
        print(f"TRAINING YOLO11{self.model_size.upper()} HURRICANE DETECTOR")
        print("="*80)
        print(f"Pretrained weights: {self.pretrained_weights}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch}")
        print(f"Device: {device}")
        print(f"Data augmentation: {augment}")
        print("="*80)
        print()

        # Load pretrained model
        model = YOLO(self.pretrained_weights)

        # Training hyperparameters
        train_args = {
            'data': str(self.data_yaml),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'patience': patience,
            'save': True,
            'save_period': save_period,
            'project': str(self.output_dir),
            'name': f'yolo11{self.model_size}_hurricane_{datetime.now().strftime("%Y%m%d_%H%M")}',
            'exist_ok': True,
            'verbose': verbose,
            'plots': True,  # Save training plots
            'val': True,    # Validate during training
        }

        # Data augmentation settings (if enabled)
        if augment:
            train_args.update({
                'hsv_h': 0.015,       # Hue augmentation
                'hsv_s': 0.7,         # Saturation augmentation
                'hsv_v': 0.4,         # Value augmentation
                'degrees': 10,        # Rotation (±10°)
                'translate': 0.1,     # Translation (10% of image)
                'scale': 0.5,         # Scaling (50-150%)
                'mosaic': 1.0,        # Mosaic augmentation probability
                'mixup': 0.1,         # Mixup augmentation probability
                'flipud': 0.0,        # No vertical flip (not meaningful for hurricanes)
                'fliplr': 0.0         # No horizontal flip (preserve rotation direction)
            })

        # Train model
        print("\nStarting training...")
        start_time = datetime.now()

        results = model.train(**train_args)

        elapsed = (datetime.now() - start_time).total_seconds()

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Elapsed time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print("="*80)
        print()

        # Return results
        return {
            'results': results,
            'model': model,
            'elapsed_seconds': elapsed,
            'output_dir': train_args['project'] + '/' + train_args['name']
        }

    def evaluate(self, model_path: str = None, model=None) -> dict:
        """
        Evaluate trained model on validation set.

        Args:
            model_path: Path to trained model weights (.pt file)
            model: Trained YOLO model object (if already loaded)

        Returns:
            Dictionary with evaluation metrics
        """
        if model is None and model_path is None:
            raise ValueError("Either model_path or model must be provided")

        if model is None:
            model = YOLO(model_path)

        print("="*80)
        print("EVALUATING MODEL")
        print("="*80)
        print()

        # Run validation
        metrics = model.val()

        # Extract metrics
        results = {
            'mAP50': float(metrics.box.map50),        # mAP at IoU=0.5
            'mAP50-95': float(metrics.box.map),       # mAP at IoU=0.5:0.95
            'precision': float(metrics.box.mp),       # Mean precision
            'recall': float(metrics.box.mr),          # Mean recall
            'fitness': float(metrics.fitness)         # Combined fitness score
        }

        print("Evaluation Results:")
        print(f"  mAP@50: {results['mAP50']:.4f} ({results['mAP50']*100:.1f}%)")
        print(f"  mAP@50-95: {results['mAP50-95']:.4f} ({results['mAP50-95']*100:.1f}%)")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  Fitness: {results['fitness']:.4f}")
        print()

        # Quality assessment
        if results['mAP50'] >= 0.75:
            print("✅ Model quality: GOOD (mAP@50 ≥ 75%)")
            print("   Ready for auto-annotation")
        elif results['mAP50'] >= 0.60:
            print("⚠️  Model quality: ACCEPTABLE (mAP@50 60-75%)")
            print("   Review auto-annotations carefully")
        else:
            print("❌ Model quality: POOR (mAP@50 < 60%)")
            print("   Not recommended for auto-annotation")
            print("   Consider: more training data, larger model, or longer training")

        print("="*80)
        print()

        return results

    def export_model(self, model_path: str, export_format: str = 'onnx',
                    output_name: str = None) -> str:
        """
        Export model to deployment format.

        Args:
            model_path: Path to trained .pt model
            export_format: Export format (onnx, torchscript, coreml, etc.)
            output_name: Custom output filename

        Returns:
            Path to exported model
        """
        model = YOLO(model_path)

        print(f"Exporting model to {export_format} format...")
        export_path = model.export(format=export_format)

        print(f"✅ Model exported: {export_path}")

        return export_path


def train_quick_test(model_size: str = 'n', epochs: int = 10):
    """
    Quick training test with minimal epochs for debugging.

    Args:
        model_size: YOLO model size
        epochs: Number of epochs (default 10 for quick test)
    """
    print("="*80)
    print("QUICK TEST MODE (Minimal epochs for debugging)")
    print("="*80)
    print()

    trainer = YOLOHurricaneTrainer(model_size=model_size)
    trainer.verify_dataset()

    result = trainer.train(
        epochs=epochs,
        batch=8,      # Smaller batch for testing
        device='mps',
        patience=5,
        save_period=5
    )

    # Evaluate
    trainer.evaluate(model=result['model'])

    print(f"\nModel saved to: {result['output_dir']}")
    print("To use for auto-annotation, find the best.pt file in the output directory")


def train_full(model_size: str = 'n', epochs: int = 100, device: str = 'mps'):
    """
    Full training run for production model.

    Args:
        model_size: YOLO model size (n, s, m, l, x)
        epochs: Number of training epochs
        device: Training device
    """
    print("="*80)
    print("FULL TRAINING MODE")
    print("="*80)
    print()

    trainer = YOLOHurricaneTrainer(model_size=model_size)
    stats = trainer.verify_dataset()

    # Verify sufficient data
    if stats['train_images'] < 100:
        print(f"⚠️  Warning: Only {stats['train_images']} training images.")
        print("   YOLO typically needs 500+ images for good performance.")
        print("   Results may be suboptimal.")
        print()

    result = trainer.train(
        epochs=epochs,
        batch=16,
        device=device,
        patience=20,
        save_period=10,
        augment=True
    )

    # Evaluate
    metrics = trainer.evaluate(model=result['model'])

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Model: YOLO11{model_size.upper()}")
    print(f"Training time: {result['elapsed_seconds']/60:.1f} minutes")
    print(f"mAP@50: {metrics['mAP50']*100:.1f}%")
    print(f"Output: {result['output_dir']}")
    print("="*80)

    return result, metrics


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Train YOLO11 hurricane detector')

    parser.add_argument('--mode', choices=['test', 'full'], default='full',
                       help='Training mode: test (10 epochs) or full (100 epochs)')

    parser.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge')

    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides mode default)')

    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps',
                       help='Training device')

    parser.add_argument('--data', type=str, default='data/raw/Hurricane.v3i.yolov8/data.yaml',
                       help='Path to dataset YAML')

    args = parser.parse_args()

    if args.mode == 'test':
        epochs = args.epochs if args.epochs else 10
        train_quick_test(model_size=args.model, epochs=epochs)
    else:
        epochs = args.epochs if args.epochs else 100
        train_full(model_size=args.model, epochs=epochs, device=args.device)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No arguments - show help
        print("YOLO11 Hurricane Detector Training")
        print("="*80)
        print()
        print("Usage:")
        print("  python train_yolo11.py --mode test      # Quick 10-epoch test")
        print("  python train_yolo11.py --mode full      # Full 100-epoch training")
        print("  python train_yolo11.py --model s        # Use small model")
        print("  python train_yolo11.py --epochs 50      # Custom epoch count")
        print()
        print("For more options, run: python train_yolo11.py --help")
        print()
        print("="*80)
        print()
        print("Running quick test by default...")
        print()
        train_quick_test(model_size='n', epochs=5)
    else:
        main()
