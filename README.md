# Hurricane Storm Forecasting Pipeline

Machine learning pipeline for predicting hurricane position, size, and intensity from thermal satellite imagery.

## Project Status

**Phase 1: Foundation** ✅ In Progress

- ✅ Pipeline design completed (see CLAUDE.md)
- ✅ Project structure created
- ✅ Preprocessing scripts ready
- ✅ Storm tracking algorithm designed
- ⏳ Awaiting data analysis review

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing Pipeline

```bash
# Step 1: Combine train/valid/test datasets
python preprocessing/1_combine_datasets.py

# Step 2: Extract YOLO spatial features
python preprocessing/2_extract_yolo_features.py

# Step 3: Extract thermal features from images
python preprocessing/3_extract_thermal_features.py

# Step 4: Build temporal sequences
python preprocessing/4_build_sequences.py

# Step 5: Track storms across frames (after reviewing analysis)
python preprocessing/5_track_storms.py  # To be implemented
```

## Project Structure

```
GK-2A/
├── CLAUDE.md                          # Complete pipeline specifications
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── raw/
│   │   ├── Hurricane.v3i.yolov8/      # Original YOLO dataset
│   │   └── combined/                  # Merged train+val+test
│   ├── processed/
│   │   ├── sequences/                 # Temporal sequences
│   │   ├── features/                  # Extracted features
│   │   └── storm_tracking/            # Storm tracking data
│   └── splits/                        # Train/test splits
│
├── preprocessing/                     # Data processing scripts
│   ├── 1_combine_datasets.py
│   ├── 2_extract_yolo_features.py
│   ├── 3_extract_thermal_features.py
│   ├── 4_build_sequences.py
│   └── 5_track_storms.py
│
├── analysis/                          # Data analysis & reports
│   ├── reports/
│   │   ├── combined_data_report.md
│   │   └── storm_tracking_report.md
│   └── storm_tracker.py               # Storm tracking implementation
│
└── [Other modules to be built...]
```

## Key Findings from Analysis

### Combined Dataset Analysis
- **822 total images** (vs 576 train-only)
- **99.6% temporal continuity** (vs 71.3%)
- **4 continuous sequences** (vs 122 fragmented)
- **Longest sequence: 573 frames** spanning ~4 days

### Storm Tracking
- **Hungarian algorithm recommended** with 100px distance threshold
- **75 storm tracks** identified (≥3 frames)
- **Longest track: 206 frames** (34+ hours)
- Storm tracking code ready in `storm_tracker.py`

## Next Steps

1. **Review Analysis Reports** (Current)
   - `analysis/reports/combined_data_report.md`
   - `analysis/reports/storm_tracking_report.md`

2. **Run Preprocessing Pipeline**
   - Combine datasets
   - Extract features
   - Build sequences

3. **Implement Storm Tracking**
   - Integrate `storm_tracker.py` into preprocessing
   - Generate tracked storm dataset

4. **Feature Engineering**
   - Motion features (velocity, acceleration)
   - Temporal delta features
   - Feature metadata system

5. **PCA Analysis**
   - Grouped PCA (thermal + spatial)
   - Second derivative elbow detection

6. **Training Pipeline**
   - Gradio GUI
   - RF, XGBoost, LightGBM models
   - Multi-horizon forecasting (t+1, t+3, t+6, t+12)

7. **Evaluation**
   - Metrics calculation
   - Trajectory visualization
   - Performance reports

## Documentation

- **CLAUDE.md** - Complete pipeline specifications
- **dataset_analysis_report.md** - Initial train-only analysis
- **analysis/reports/combined_data_report.md** - Combined dataset analysis
- **analysis/reports/storm_tracking_report.md** - Storm tracking algorithm

## Performance Targets

**Phase 1 Goals:**
- t+1 (10 min): Test R² > 0.75 for position
- t+3 (30 min): Test R² > 0.60 for position
- t+6 (1 hour): Test R² > 0.45 for position
- Training time: <5 min per model

## Contact & Credits

GOES-18 satellite imagery from NOAA
Dataset: Hurricane v3i (YOLOv8 format)
