"""
Step 1: Zero-Shot Inference Test
Test pre-trained TGE-YOLO (typhoon) model on hurricane dataset without training.
"""

import os
import sys
from pathlib import Path

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
    print("TGE-YOLO STEP 1: Zero-Shot Inference Test")
    print("=" * 80)
    print("\n📖 Testing typhoon-trained model on hurricane data (no training)")

    # Paths
    model_path = project_root / 'docs/TGE-YOLO/ultralytics-main/runs/detect/train10/weights/best.pt'
    data_yaml = project_root / 'data/raw/Hurricane.v3i.yolov8/data.yaml'
    results_dir = project_root / 'tge_yolo_evaluation/results'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Model: {model_path.name}")
    print(f"📊 Dataset: Hurricane.v3i.yolov8")
    print(f"🎯 Task: Object detection (hurricane localization)")

    # Load model
    print(f"\n⏳ Loading pre-trained TGE-YOLO model...")
    try:
        model = YOLO(str(model_path))
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Model info
    print(f"\n🔍 Model Details:")
    print(f"   Architecture: TGE-YOLO")
    print(f"   Innovations:")
    print(f"      - TFAM_Concat: Feature fusion with dual attention")
    print(f"      - GSConv: Grouped shuffle convolution (-9.3% params)")
    print(f"      - E-EIoU Loss: Enhanced center localization")
    print(f"   Training Data: 11,976 typhoon images (Himawari satellite)")

    # Run validation
    print(f"\n🧪 Running validation on hurricane TEST set...")
    print(f"   (This may take a few minutes)")

    try:
        metrics = model.val(
            data=str(data_yaml),
            split='test',
            imgsz=640,
            batch=16,
            verbose=False,
            plots=True,
            save_json=True,
            project=str(results_dir),
            name='zero_shot'
        )

        # Display results
        print("\n" + "=" * 80)
        print("📊 ZERO-SHOT RESULTS: Typhoon Model → Hurricane Data")
        print("=" * 80)

        print(f"\n🎯 Detection Performance:")
        print(f"   mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"   mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"   Precision    : {metrics.box.p:.4f}")
        print(f"   Recall       : {metrics.box.r:.4f}")

        # Interpretation
        print(f"\n💡 Interpretation:")
        if metrics.box.map50 > 0.75:
            print(f"   ✅ EXCELLENT transfer (mAP > 0.75)")
            print(f"      Typhoon and hurricane features are very similar!")
            print(f"      Fine-tuning will likely push mAP > 0.85")
        elif metrics.box.map50 > 0.60:
            print(f"   ✅ GOOD transfer (mAP > 0.60)")
            print(f"      Model captures general storm patterns")
            print(f"      Fine-tuning recommended to adapt to hurricanes")
        elif metrics.box.map50 > 0.40:
            print(f"   ⚠️  MODERATE transfer (mAP > 0.40)")
            print(f"      Some domain gap between typhoons/hurricanes")
            print(f"      Transfer learning will help significantly")
        else:
            print(f"   ❌ POOR transfer (mAP < 0.40)")
            print(f"      Large domain gap detected")
            print(f"      May need training from scratch")

        # Save results
        results_file = results_dir / 'zero_shot_metrics.txt'
        with open(results_file, 'w') as f:
            f.write("TGE-YOLO Zero-Shot Inference Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: Hurricane.v3i.yolov8 (GOES-18)\n")
            f.write(f"Training: None (zero-shot)\n\n")
            f.write("METRICS:\n")
            f.write(f"  mAP@0.5      : {metrics.box.map50:.4f}\n")
            f.write(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}\n")
            f.write(f"  Precision    : {metrics.box.p:.4f}\n")
            f.write(f"  Recall       : {metrics.box.r:.4f}\n\n")
            f.write("CONTEXT:\n")
            f.write("  - Pre-trained on 11,976 typhoon images (Himawari)\n")
            f.write("  - Tested on hurricane data without fine-tuning\n")
            f.write("  - Measures typhoon → hurricane transfer learning\n")

        print(f"\n💾 Results saved:")
        print(f"   Metrics: {results_file}")
        print(f"   Plots: {results_dir / 'zero_shot'}")

        # Next steps
        print("\n" + "=" * 80)
        print("📋 NEXT STEPS:")
        print("=" * 80)
        print("\n🎯 Step 2: Transfer Learning (Recommended)")
        print("   └─ Fine-tune this model on hurricane data")
        print("   └─ Expected improvement: +5-15% mAP")
        print("   └─ Run: python scripts/02_transfer_learning.py")
        print("\n🔥 Step 3: Train from Scratch (Baseline)")
        print("   └─ Train TGE-YOLO architecture from random weights")
        print("   └─ For comparison: does transfer learning help?")
        print("   └─ Run: python scripts/03_train_from_scratch.py")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✨ Zero-shot inference complete!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
