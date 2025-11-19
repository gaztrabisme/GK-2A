# Hurricane Storm Forecasting Pipeline

Machine learning pipeline for predicting hurricane position, size, and intensity from thermal satellite imagery.

## Project Status

**Phase 1: Foundation** ✅ **COMPLETE**

- ✅ Pipeline design completed (see CLAUDE.md)
- ✅ Project structure created
- ✅ Preprocessing pipeline implemented
- ✅ Storm tracking algorithm implemented
- ✅ Feature engineering complete (spatial, thermal, motion, temporal)
- ✅ PCA analysis with auto-selection
- ✅ Training pipeline (RF, XGBoost, LightGBM + Stacking)
- ✅ Model evaluation & metrics
- ✅ Interactive visualization GUI

## 🎯 Achieved Results

**Model Performance** (LightGBM Stacking Ensemble):
- ✅ **t+1 (10 min)**: R² = 0.862 (Target: >0.75)
- ✅ **t+3 (30 min)**: R² = 0.817 (Target: >0.60)
- ✅ **t+6 (1 hour)**: R² = 0.761 (Target: >0.45)
- ✅ **t+12 (2 hours)**: R² = 0.595 (LightGBM baseline)

**All Phase 1 targets exceeded!**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. View Interactive Visualization 🎨

**Launch the forecast visualization GUI:**

```bash
python visualization/gradio_app.py
```

This opens an interactive interface showing:
- Zoomable satellite imagery (Oct 17-21, 2023)
- Real-time hurricane predictions with confidence scores
- Actual vs predicted trajectories
- Position error metrics (% of image ≈ 75-100 km on full Earth disk)

**Share publicly:**
- The app automatically generates a public URL (Gradio share)
- Or use ngrok for custom domain (see CLAUDE.md section 12)

### 3. Train Models

```bash
# Complete training pipeline
python training/train_all_models.py
```

### 4. Preprocessing Pipeline (Already Complete)

```bash
# Step 1: Combine train/valid/test datasets
python preprocessing/1_combine_datasets.py

# Step 2: Extract YOLO spatial + thermal features
python preprocessing/2_extract_yolo_features.py

# Step 3: Build temporal sequences
python preprocessing/3_build_sequences.py

# Step 4: Track storms across frames
python preprocessing/4_track_storms.py
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
│   │   └── Hurricane.v3i.yolov8/      # Original YOLO dataset
│   ├── processed/
│   │   ├── sequences/                 # Temporal sequences
│   │   ├── features/                  # Extracted features (spatial, thermal, motion, temporal)
│   │   └── storm_tracking/            # Storm tracking data
│   └── splits/                        # Train/test splits
│
├── preprocessing/                     # Data processing scripts
│   ├── 1_combine_datasets.py          # Merge train/val/test
│   ├── 2_extract_yolo_features.py     # YOLO + thermal extraction
│   ├── 3_build_sequences.py           # Temporal sequence building
│   └── 4_track_storms.py              # Hungarian algorithm tracking
│
├── features/                          # Feature engineering modules
│   ├── metadata.yml                   # Feature definitions
│   ├── core/
│   │   └── sequence_api.py            # Central data access
│   ├── spatial.py                     # Bbox-derived features
│   ├── thermal.py                     # Color-based features
│   ├── motion.py                      # Velocity, acceleration
│   └── temporal.py                    # Delta features
│
├── pca/                               # PCA analysis
│   ├── pca_analyzer.py                # Grouped PCA with elbow detection
│   ├── config/
│   │   └── pca_config.yml             # Auto-generated PC selections
│   └── transformers/                  # Fitted PCA & scalers
│
├── training/                          # Training pipeline
│   ├── train_all_models.py            # Main training script
│   ├── models/                        # Model implementations
│   ├── splits/                        # Split strategies
│   └── trained_models/                # Saved models (RF, XGB, LGBM, stacking)
│
├── evaluation/                        # Model evaluation
│   ├── metrics.py                     # RMSE, MAE, R² calculations
│   └── reports/                       # Performance reports
│
├── visualization/                     # Interactive GUI
│   ├── gradio_app.py                  # Gradio + Plotly interface
│   └── forecast_viz.py                # Data loader for visualization
│
└── analysis/                          # Analysis & reports
    └── reports/
        ├── combined_data_report.md
        └── storm_tracking_report.md
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

## Features

### 🎨 Interactive Visualization
- **Zoomable satellite imagery** with Plotly pan/zoom controls
- **Multi-horizon forecasts**: 10min, 30min, 1hr, 2hrs ahead
- **Real-time error metrics**: Position offset % (1% ≈ 75-100 km)
- **Color-coded trajectories**:
  - Magenta: Current positions (YOLO)
  - White: Ground truth paths
  - Purple/Green/Pink/Cyan: Predictions by horizon
- **Timeline navigation**: 642 frames (Oct 17-21, 2023)
- **Public sharing**: Built-in Gradio share or ngrok

### 🧠 Machine Learning Pipeline
- **Ensemble stacking**: LightGBM meta-model on RF + XGBoost + LightGBM
- **Multi-target regression**: Position (x, y), size, intensity
- **Multi-horizon forecasting**: t+1, t+3, t+6, t+12
- **Feature engineering**: 29 features across spatial, thermal, motion, temporal groups
- **PCA dimensionality reduction**: Auto-selection via elbow detection
- **Sequence-based temporal split**: Prevents data leakage

### 📊 Data Processing
- **Storm tracking**: Hungarian algorithm with 100px threshold
- **Temporal sequences**: 4 continuous sequences, 99.6% continuity
- **75 tracked storms**: Longest track 206 frames (34+ hours)
- **2410 total samples**: 490 train, 1920 test

## Next Steps (Phase 2)

1. **LSTM Implementation**
   - Leverage temporal sequences for RNN models
   - Target: Outperform tree models on t+6, t+12

2. **Real-time Inference**
   - Deploy API for live predictions
   - Target: <100ms per prediction

3. **Advanced Features**
   - Storm evolution patterns
   - Environmental context (sea surface temp, wind shear)
   - Multi-modal satellite channels

4. **Extended Forecasting**
   - 6-hour, 12-hour, 24-hour horizons
   - Uncertainty quantification

## Documentation

- **CLAUDE.md** - Complete pipeline specifications (updated with visualization section)
- **README.md** - This file (project overview)
- **analysis/reports/combined_data_report.md** - Combined dataset analysis
- **analysis/reports/storm_tracking_report.md** - Storm tracking algorithm
- **evaluation/reports/** - Model performance reports

## Technologies Used

**Machine Learning**:
- scikit-learn (Random Forest, preprocessing)
- XGBoost (gradient boosting)
- LightGBM (high-performance GBDT + stacking)
- NumPy, Pandas (data processing)

**Visualization**:
- Gradio (web interface)
- Plotly (interactive plots)
- OpenCV (image processing)
- Matplotlib (static plots)

**Data**:
- GOES-18 satellite imagery (NOAA)
- YOLOv8 format annotations
- Hurricane v3i dataset

## Contact & Credits

- **Satellite Data**: GOES-18 ABI Full Disk Sandwich from NOAA
- **Dataset**: Hurricane v3i (YOLOv8 format)
- **Models**: Random Forest, XGBoost, LightGBM with stacking ensemble

---

*Phase 1 Complete - All targets exceeded ✅*
