"""
Step 2: Transfer Learning
Fine-tune pre-trained TGE-YOLO (typhoon model) on hurricane dataset.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add TGE-YOLO ultralytics to path
tge_yolo_path = project_root / 'docs/TGE-YOLO/ultralytics-main'
sys.path.insert(0, str(tge_yolo_path))

import torch
from ultralytics import YOLO

# Monkey-patch torch.load to allow unsafe loading for this trusted source
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

def main():
    print("=" * 80)
    print("TGE-YOLO STEP 2: Transfer Learning (Fine-tuning)")
    print("=" * 80)
    print("\n🎯 Fine-tuning typhoon-trained model on hurricane data")

    # Paths
    model_path = project_root / 'docs/TGE-YOLO/ultralytics-main/runs/detect/train10/weights/best.pt'
    data_yaml = project_root / 'data/raw/Hurricane.v3i.yolov8/data.yaml'
    results_dir = project_root / 'tge_yolo_evaluation/results'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Starting Model: {model_path.name}")
    print(f"   - Pre-trained on 11,976 typhoon images")
    print(f"   - Original mAP: 87.8% (on typhoons)")
    print(f"   - Zero-shot mAP: 0.0% (on hurricanes)")
    print(f"\n📊 Training Dataset: Hurricane.v3i.yolov8")
    print(f"   - Train: 576 images")
    print(f"   - Valid: 123 images")
    print(f"   - Test: 123 images")

    # Training configuration
    config = {
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'patience': 20,  # Early stopping
        'save_period': 10,  # Save checkpoint every 10 epochs
        'device': 'cpu',  # M3 Max CPU
        'workers': 4,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Lower learning rate for fine-tuning
        'lrf': 0.01,  # Final learning rate factor
        'warmup_epochs': 3,
        'weight_decay': 0.0005,
        'dropout': 0.0,
        'verbose': True,
        'plots': True,
        'save_json': True,
        # Disable wandb and other tracking tools to avoid errors
        'project': 'tge_yolo_hurricane',  # Simple name without slashes
        'name': 'transfer_learning',
    }

    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {config['epochs']} (with early stopping patience={config['patience']})")
    print(f"   Batch Size: {config['batch']}")
    print(f"   Image Size: {config['imgsz']}x{config['imgsz']}")
    print(f"   Learning Rate: {config['lr0']} → {config['lr0'] * config['lrf']}")
    print(f"   Optimizer: {config['optimizer']}")
    print(f"   Device: {config['device']}")

    # Load pre-trained model
    print(f"\n⏳ Loading pre-trained TGE-YOLO model...")
    try:
        model = YOLO(str(model_path))
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Start training
    print(f"\n🔥 Starting transfer learning...")
    print(f"   (This may take 1-2 hours on CPU)")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-" * 80)

    start_time = datetime.now()

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=config['epochs'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            patience=config['patience'],
            save_period=config['save_period'],
            device=config['device'],
            workers=config['workers'],
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            warmup_epochs=config['warmup_epochs'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            verbose=config['verbose'],
            plots=config['plots'],
            save_json=config['save_json'],
            project=config['project'],
            name=config['name']
        )

        end_time = datetime.now()
        training_duration = end_time - start_time

        print("\n" + "=" * 80)
        print("✅ TRANSFER LEARNING COMPLETE!")
        print("=" * 80)
        print(f"\n⏱️  Training Duration: {training_duration}")
        print(f"   Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Get best model path (YOLO saves in project/name by default)
        best_model_path = Path(config['project']) / config['name'] / 'weights' / 'best.pt'

        # Run validation on test set with best model
        print(f"\n🧪 Validating best model on TEST set...")
        best_model = YOLO(str(best_model_path))

        metrics = best_model.val(
            data=str(data_yaml),
            split='test',
            imgsz=640,
            batch=16,
            verbose=False,
            plots=True,
            save_json=True,
            project=config['project'],
            name='test_evaluation'
        )

        # Display results
        print("\n" + "=" * 80)
        print("📊 TRANSFER LEARNING RESULTS")
        print("=" * 80)

        print(f"\n🎯 Test Set Performance:")
        print(f"   mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"   mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"   Precision    : {metrics.box.mp:.4f}")
        print(f"   Recall       : {metrics.box.mr:.4f}")

        # Comparison with zero-shot
        print(f"\n📈 Improvement over Zero-Shot:")
        print(f"   Zero-shot mAP@0.5: 0.0000")
        print(f"   Transfer mAP@0.5:  {metrics.box.map50:.4f}")
        if metrics.box.map50 > 0:
            print(f"   ✅ Transfer learning successful!")
        else:
            print(f"   ⚠️  Still no detections - may need more training")

        # Comparison with original typhoon performance
        original_map = 0.878
        print(f"\n🌪️  Comparison with Original (Typhoon):")
        print(f"   Typhoon mAP@0.5:   {original_map:.4f}")
        print(f"   Hurricane mAP@0.5: {metrics.box.map50:.4f}")
        if metrics.box.map50 > 0:
            retention = (metrics.box.map50 / original_map) * 100
            print(f"   Performance retention: {retention:.1f}%")

        # Save detailed results
        results_file = results_dir / 'transfer_learning_metrics.txt'
        with open(results_file, 'w') as f:
            f.write("TGE-YOLO Transfer Learning Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Duration: {training_duration}\n")
            f.write(f"Epochs Completed: {results.epoch + 1}\n")
            f.write(f"Best Epoch: {results.best_epoch}\n\n")
            f.write("CONFIGURATION:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nTEST SET METRICS:\n")
            f.write(f"  mAP@0.5      : {metrics.box.map50:.4f}\n")
            f.write(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}\n")
            f.write(f"  Precision    : {metrics.box.mp:.4f}\n")
            f.write(f"  Recall       : {metrics.box.mr:.4f}\n\n")
            f.write("IMPROVEMENT:\n")
            f.write(f"  Zero-shot → Transfer: +{metrics.box.map50:.4f} mAP\n")

        print(f"\n💾 Results saved:")
        print(f"   Metrics: {results_file}")
        print(f"   Best Model: {best_model_path}")
        print(f"   Training Plots: {Path(config['project']) / config['name']}")
        print(f"   Test Predictions: {Path(config['project']) / 'test_evaluation'}")

        # Next steps
        print("\n" + "=" * 80)
        print("📋 NEXT STEPS:")
        print("=" * 80)
        if metrics.box.map50 > 0.60:
            print("\n✅ Transfer learning was successful!")
            print("   Proceed to Step 3 for comparison baseline:")
            print("   └─ python scripts/03_train_from_scratch.py")
        elif metrics.box.map50 > 0.40:
            print("\n⚠️  Moderate performance achieved.")
            print("   Consider:")
            print("   1. Training for more epochs")
            print("   2. Adjusting hyperparameters")
            print("   3. Comparing with training from scratch (Step 3)")
        else:
            print("\n⚠️  Poor transfer performance.")
            print("   Recommendations:")
            print("   1. Try training from scratch (Step 3)")
            print("   2. Check data quality and labels")
            print("   3. Consider data augmentation")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✨ Transfer learning process complete!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
