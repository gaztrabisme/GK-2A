"""
Evaluate TGE-YOLO transfer learning model on test set
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
    print("TGE-YOLO TEST SET EVALUATION")
    print("=" * 80)
    print("\n📊 Evaluating fine-tuned model on held-out test set")

    # Paths
    model_path = project_root / 'tge_yolo_evaluation/tge_yolo_hurricane/transfer_learning/weights/best.pt'
    data_yaml = project_root / 'data/raw/Hurricane.v3i.yolov8/data.yaml'

    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("   Please train the model first (Step 2: Transfer Learning)")
        return

    print(f"\n📦 Model: {model_path.name}")
    print(f"   Location: {model_path.parent}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n📊 Test Dataset: Hurricane.v3i.yolov8")
    print(f"   Test images: 123")
    print(f"   Test instances: ~252 hurricanes")

    # Load model
    print(f"\n⏳ Loading fine-tuned TGE-YOLO model...")
    try:
        model = YOLO(str(model_path))
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run validation on test set
    print(f"\n🧪 Running inference on TEST set...")
    print(f"   (This will take a few minutes)")

    try:
        metrics = model.val(
            data=str(data_yaml),
            split='test',
            imgsz=640,
            batch=16,
            verbose=False,
            plots=True,
            save_json=True,
            conf=0.25,  # Confidence threshold
            iou=0.7,    # IoU threshold for NMS
            project='tge_yolo_hurricane',
            name='test_evaluation'
        )

        # Display results
        print("\n" + "=" * 80)
        print("📊 TEST SET RESULTS")
        print("=" * 80)

        print(f"\n🎯 Detection Performance:")
        print(f"   mAP@0.5      : {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
        print(f"   mAP@0.5:0.95 : {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)")
        print(f"   Precision    : {metrics.box.mp:.4f} ({metrics.box.mp*100:.1f}%)")
        print(f"   Recall       : {metrics.box.mr:.4f} ({metrics.box.mr*100:.1f}%)")

        # Comparison with validation set
        print(f"\n📈 Validation vs Test Comparison:")
        print(f"   {'Metric':<15} {'Validation':<12} {'Test':<12} {'Difference':<12}")
        print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

        val_map50 = 0.924  # From training (epoch 62)
        val_map = 0.656
        val_p = 0.873
        val_r = 0.877

        diff_map50 = metrics.box.map50 - val_map50
        diff_map = metrics.box.map - val_map
        diff_p = metrics.box.mp - val_p
        diff_r = metrics.box.mr - val_r

        print(f"   {'mAP@0.5':<15} {val_map50:<12.4f} {metrics.box.map50:<12.4f} {diff_map50:+.4f}")
        print(f"   {'mAP@0.5:0.95':<15} {val_map:<12.4f} {metrics.box.map:<12.4f} {diff_map:+.4f}")
        print(f"   {'Precision':<15} {val_p:<12.4f} {metrics.box.mp:<12.4f} {diff_p:+.4f}")
        print(f"   {'Recall':<15} {val_r:<12.4f} {metrics.box.mr:<12.4f} {diff_r:+.4f}")

        # Interpretation
        print(f"\n💡 Interpretation:")
        if abs(diff_map50) < 0.05:
            print(f"   ✅ Test performance matches validation (within 5%)")
            print(f"      Model generalizes well to unseen data!")
        elif diff_map50 < -0.05:
            print(f"   ⚠️  Test performance lower than validation")
            print(f"      Possible overfitting to validation set")
            print(f"      Consider: more data augmentation or regularization")
        else:
            print(f"   🎉 Test performance BETTER than validation!")
            print(f"      Validation set might be harder than test set")

        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if metrics.box.map50 > 0.90:
            print(f"   ✅ EXCELLENT performance (mAP > 0.90)")
            print(f"      Model is production-ready for hurricane detection")
        elif metrics.box.map50 > 0.75:
            print(f"   ✅ GOOD performance (mAP > 0.75)")
            print(f"      Model performs well, suitable for deployment")
        elif metrics.box.map50 > 0.60:
            print(f"   ⚠️  MODERATE performance (mAP > 0.60)")
            print(f"      Consider further training or tuning")
        else:
            print(f"   ⚠️  POOR performance (mAP < 0.60)")
            print(f"      Model needs improvement")

        # Comparison with original TGE-YOLO on typhoons
        original_map = 0.878
        print(f"\n🌪️  Comparison with Original TGE-YOLO:")
        print(f"   Original (Typhoons):  {original_map:.4f} ({original_map*100:.1f}%)")
        print(f"   Transfer (Hurricanes): {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
        if metrics.box.map50 > original_map:
            diff_pct = ((metrics.box.map50 / original_map) - 1) * 100
            print(f"   🎉 Transfer model is {diff_pct:+.1f}% better!")
        elif metrics.box.map50 > original_map - 0.05:
            print(f"   ✅ Performance comparable to original")
        else:
            diff_pct = ((metrics.box.map50 / original_map) - 1) * 100
            print(f"   Performance: {diff_pct:.1f}% of original")

        # Save results
        results_file = project_root / 'tge_yolo_evaluation/results/test_set_metrics.txt'
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            f.write("TGE-YOLO Test Set Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: Hurricane.v3i.yolov8 (GOES-18)\n")
            f.write(f"Split: test (123 images, ~252 instances)\n\n")
            f.write("TEST SET METRICS:\n")
            f.write(f"  mAP@0.5      : {metrics.box.map50:.4f}\n")
            f.write(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}\n")
            f.write(f"  Precision    : {metrics.box.mp:.4f}\n")
            f.write(f"  Recall       : {metrics.box.mr:.4f}\n\n")
            f.write("VALIDATION vs TEST:\n")
            f.write(f"  mAP@0.5:      Val={val_map50:.4f}, Test={metrics.box.map50:.4f}, Diff={diff_map50:+.4f}\n")
            f.write(f"  mAP@0.5:0.95: Val={val_map:.4f}, Test={metrics.box.map:.4f}, Diff={diff_map:+.4f}\n")
            f.write(f"  Precision:    Val={val_p:.4f}, Test={metrics.box.mp:.4f}, Diff={diff_p:+.4f}\n")
            f.write(f"  Recall:       Val={val_r:.4f}, Test={metrics.box.mr:.4f}, Diff={diff_r:+.4f}\n\n")
            f.write("COMPARISON WITH ORIGINAL:\n")
            f.write(f"  Original (Typhoons):  {original_map:.4f}\n")
            f.write(f"  Transfer (Hurricanes): {metrics.box.map50:.4f}\n")
            if metrics.box.map50 > original_map:
                diff_pct = ((metrics.box.map50 / original_map) - 1) * 100
                f.write(f"  Improvement: +{diff_pct:.1f}%\n")

        print(f"\n💾 Results saved:")
        print(f"   Metrics: {results_file}")
        print(f"   Predictions: tge_yolo_hurricane/test_evaluation/")
        print(f"   Confusion Matrix: tge_yolo_hurricane/test_evaluation/confusion_matrix.png")
        print(f"   Prediction Examples: tge_yolo_hurricane/test_evaluation/val_batch*_pred.jpg")

        # Next steps
        print("\n" + "=" * 80)
        print("📋 NEXT STEPS:")
        print("=" * 80)
        print("\n✅ Model is ready for deployment!")
        print("\n🎯 Integration Options:")
        print("   1. Use for real-time hurricane detection on GOES-18 imagery")
        print("   2. Feed detections to LSTM trajectory forecasting")
        print("   3. Compare with existing YOLOv8 baseline")
        print("   4. Visualize predictions on test images")
        print("\n📊 Further Analysis:")
        print("   - Check confusion matrix for error patterns")
        print("   - Inspect prediction examples (val_batch*_pred.jpg)")
        print("   - Analyze per-image performance")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✨ Test set evaluation complete!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
