"""
Quick Inference Test: TGE-YOLO Pre-trained Model on Hurricane Dataset

Tests the typhoon-trained TGE-YOLO model on our hurricane test set.
"""

import sys
from pathlib import Path

# Add TGE-YOLO ultralytics to path
tge_yolo_path = Path('docs/TGE-YOLO/ultralytics-main')
sys.path.insert(0, str(tge_yolo_path))

from ultralytics import YOLO

def main():
    print("=" * 80)
    print("TGE-YOLO Inference Test - Step 1: Zero-shot Evaluation")
    print("=" * 80)

    # Paths
    model_path = 'docs/TGE-YOLO/ultralytics-main/runs/detect/train10/weights/best.pt'
    data_yaml = 'data/raw/Hurricane.v3i.yolov8/data.yaml'

    print(f"\n📦 Loading pre-trained TGE-YOLO model...")
    print(f"   Model: {model_path}")

    try:
        # Load with weights_only=False to allow custom ultralytics classes
        # This is safe since we trust the source (research paper authors)
        import torch
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

        model = YOLO(model_path)
        print("   ✅ Model loaded successfully!")

        # Print model info
        print(f"\n🔍 Model Architecture:")
        print(f"   - Type: TGE-YOLO (GSConv + TFAM)")
        print(f"   - Trained on: ~12K typhoon images (Himawari satellite)")

    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # Run validation on hurricane test set
    print(f"\n🧪 Running validation on Hurricane test set...")
    print(f"   Dataset: {data_yaml}")

    try:
        metrics = model.val(
            data=data_yaml,
            split='test',
            imgsz=640,
            batch=16,
            verbose=True
        )

        print("\n" + "=" * 80)
        print("📊 RESULTS: Zero-shot TGE-YOLO on Hurricanes")
        print("=" * 80)

        print(f"\n🎯 Detection Metrics:")
        print(f"   mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"   mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"   Precision    : {metrics.box.p:.4f}")
        print(f"   Recall       : {metrics.box.r:.4f}")

        print(f"\n📍 Per-Class Metrics:")
        if hasattr(metrics.box, 'maps'):
            for i, map_val in enumerate(metrics.box.maps):
                print(f"   Class {i} (Hurricane): mAP = {map_val:.4f}")

        print(f"\n💡 Interpretation:")
        print(f"   - This is ZERO-SHOT performance (typhoon → hurricane)")
        print(f"   - No training on hurricane data yet!")
        print(f"   - Expected: ~60-80% of optimal performance")

        # Save results
        results_file = 'evaluation/tge_yolo_zero_shot_results.txt'
        Path('evaluation').mkdir(exist_ok=True)

        with open(results_file, 'w') as f:
            f.write("TGE-YOLO Zero-Shot Inference Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: {data_yaml}\n\n")
            f.write(f"mAP@0.5      : {metrics.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95 : {metrics.box.map:.4f}\n")
            f.write(f"Precision    : {metrics.box.p:.4f}\n")
            f.write(f"Recall       : {metrics.box.r:.4f}\n")

        print(f"\n💾 Results saved to: {results_file}")

        # Suggest next steps
        print("\n" + "=" * 80)
        print("📋 NEXT STEPS:")
        print("=" * 80)
        print("\n1. ✅ If mAP > 0.70: Model transfers well! Proceed to fine-tuning.")
        print("2. ⚠️  If mAP 0.50-0.70: Decent transfer. Fine-tuning will help significantly.")
        print("3. ❌ If mAP < 0.50: Poor transfer. May need training from scratch.")
        print("\n   Ready for Step 2 (Transfer Learning)? 🚀")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✨ Inference test complete!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
