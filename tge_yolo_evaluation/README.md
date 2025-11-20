# TGE-YOLO Evaluation on Hurricane Dataset

This folder contains evaluation of the TGE-YOLO typhoon detection model (from Scientific Reports 2025 paper) on our GK-2A hurricane forecasting dataset.

## Paper Reference
**Title**: "Typhoon localization detection algorithm based on TGE-YOLO"
**Authors**: Lan He, Ling Xiao, Peihao Yang & Sheng Li (Guangdong Ocean University)
**Published**: Scientific Reports (2025) 15:3385
**DOI**: https://doi.org/10.1038/s41598-025-87833-8

## TGE-YOLO Key Innovations
1. **TFAM_Concat**: Temporal Fusion Attention Mechanism for better feature fusion
2. **GSConv**: Grouped Shuffle Convolution for computational efficiency (-9.3% params)
3. **E-EIoU Loss**: Enhanced localization loss for precise center positioning

## Original Performance (on Typhoon Dataset)
- **mAP@0.5**: 87.8%
- **MSE**: 0.115
- **Longitude error**: 0.17°
- **Latitude error**: 0.26°
- **FPS**: 416.7

## Evaluation Strategy

### Step 1: Zero-Shot Inference ⚡ (Quick Test)
Test their pre-trained typhoon model on our hurricane data **without** training.

**Files**:
- `scripts/01_zero_shot_inference.py` - Run inference
- `results/zero_shot_metrics.txt` - Results

### Step 2: Transfer Learning 🎯 (Recommended)
Fine-tune their typhoon model on our hurricane dataset.

**Files**:
- `scripts/02_transfer_learning.py` - Fine-tuning script
- `configs/transfer_learning_config.yaml` - Training config
- `results/transfer_learning_metrics.txt` - Results

### Step 3: Train from Scratch 🔥 (Baseline Comparison)
Train TGE-YOLO architecture from scratch on hurricanes.

**Files**:
- `scripts/03_train_from_scratch.py` - Training script
- `configs/train_config.yaml` - Training config
- `results/scratch_training_metrics.txt` - Results

### Step 4: Comparison Analysis 📊
Compare all approaches with our current YOLOv8 baseline.

**Files**:
- `scripts/04_compare_results.py` - Comparison script
- `results/comparison_report.md` - Final analysis

## Folder Structure

```
tge_yolo_evaluation/
├── README.md                          # This file
├── scripts/                           # Evaluation scripts
│   ├── 01_zero_shot_inference.py
│   ├── 02_transfer_learning.py
│   ├── 03_train_from_scratch.py
│   └── 04_compare_results.py
├── configs/                           # Training configurations
│   ├── transfer_learning_config.yaml
│   └── train_config.yaml
├── results/                           # Evaluation results
│   ├── zero_shot_metrics.txt
│   ├── transfer_learning_metrics.txt
│   ├── scratch_training_metrics.txt
│   └── comparison_report.md
└── models/                            # Trained model weights
    ├── tge_yolo_finetuned.pt
    └── tge_yolo_scratch.pt
```

## Usage

### Quick Start (Step 1)
```bash
cd tge_yolo_evaluation
python scripts/01_zero_shot_inference.py
```

### Transfer Learning (Step 2)
```bash
python scripts/02_transfer_learning.py
```

### Compare Results (Step 4)
```bash
python scripts/04_compare_results.py
```

## Dataset
- **Source**: Hurricane.v3i.yolov8 (GOES-18 satellite)
- **Training**: 576 images
- **Validation**: 123 images
- **Test**: 123 images
- **Total**: 822 images (Oct 2023)

## Expected Outcomes
1. Validate if typhoon → hurricane transfer works
2. Measure improvement over baseline YOLOv8
3. Determine if TGE-YOLO modules improve hurricane detection
4. Assess impact on downstream LSTM trajectory forecasting

---

*Evaluation Date*: November 2025
