# Hurricane Storm Forecasting Pipeline

**Project Goal**: Predict future hurricane position, size, and intensity using time-series modeling on thermal satellite imagery with YOLO-detected storm bounding boxes.

---

## Pipeline Architecture

### Phase 1: Foundation & Tree Models (Current Focus)
```
1. Preprocessing → 2. Feature Engineering → 3. PCA Analysis → 4. Training (RF/XGB/LGBM) → 5. Evaluation
```

### Phase 2: Deep Learning (Future)
- LSTM models for longer-term forecasting
- Transformer-based approaches (if dataset grows)

---

## Data Specifications

### Input Data
- **Format**: YOLOv8 dataset with thermal satellite images
- **Source**: GOES-18 ABI Full Disk Sandwich (678x678 pixels)
- **Temporal Resolution**: 10-minute intervals (with gaps)
- **Annotations**: Bounding boxes for Hurricane class (Class 0)
- **Current Dataset**:
  - Train: 576 images, 122 continuous sequences, 1798 bboxes
  - Time range: Oct 15-21, 2023 (5 days, 17 hours)
  - Average: 3.14 hurricanes per image (range: 1-7)

### Data Processing Strategy
- **Combine train/val/test sets** for better temporal continuity
- Re-split after understanding combined characteristics
- Analyze combined dataset for storm tracking patterns

---

## 1. Preprocessing Pipeline

### 1.1 Data Extraction
- **YOLO Feature Extraction**: Parse bounding box coordinates, dimensions
- **Thermal Feature Extraction**: Extract pixel color values within bboxes
- **Output**: Structured data per frame (CSV/Parquet)

### 1.2 Sequence Building (Central Logic)
Core data structure:
```python
Sequence = {
    'sequence_id': 'seq_001',
    'timestamps': [t0, t1, t2, ...],
    'gap_durations': [10, 10, 20, ...],  # minutes between frames
    'storms': [
        {
            'storm_id': 'storm_A',      # tracked across frames
            'frames': [0, 1, 2, 3],     # which frames it appears in
            'bboxes': [...],            # raw YOLO data per frame
            'positions': [...],         # (x, y) per frame
            'sizes': [...],             # area per frame
            'thermal': [...]            # thermal stats per frame
        }
    ]
}
```

**Requirements**:
- Detect temporal gaps (>10 minutes)
- Split into continuous sequences (≤10 min gaps)
- Provide standardized API for feature modules

### 1.3 Storm Tracking
**Challenge**: Match same storm across frames (multi-object tracking)

**Approach**: To be determined via data analysis
- Analyze spatial proximity between frames
- Consider size/thermal similarity
- Implement Hungarian algorithm or similar
- **Action**: Deploy subagent to analyze tracking patterns in real data

---

## 2. Feature Engineering

### Modular Feature Architecture

#### 2.1 Feature Groups

**Group 1: Spatial Features** (from YOLO bboxes)
- `x_center`, `y_center` - storm position (normalized [0-1])
- `bbox_width`, `bbox_height` - storm dimensions
- `bbox_area` - derived area
- `aspect_ratio` - width/height

**Group 2: Thermal Features** (from imagery within bbox)
- `mean_color` - average pixel color
- `max_color` - maximum pixel color
- `min_color` - minimum pixel color
- `std_color` - color variation (storm intensity proxy)
- `color_gradient` - center vs. edges difference

**Group 3: Motion Features** (requires ≥2 frames)
- `velocity_x`, `velocity_y` - speed components (pixels/10min)
- `speed` - magnitude of velocity
- `direction` - angle of movement (degrees)
- `acceleration` - change in speed over time

**Group 4: Temporal Delta Features** (frame-to-frame changes)
- `delta_size` - change in bbox_area
- `delta_color_mean` - change in mean color
- `delta_color_std` - change in color variation
- `delta_speed` - change in velocity

#### 2.2 Feature Module Structure
```
features/
├── metadata.yml              # feature definitions, dependencies, groups
├── core/
│   └── sequence_builder.py   # central sequencing logic & API
├── spatial.py                # Group 1
├── thermal.py                # Group 2
├── motion.py                 # Group 3
└── temporal.py               # Group 4
```

Each feature script:
- Loads from central sequence builder
- Declares dependencies in metadata.yml
- Can be independently enabled/disabled
- Outputs standardized format

**metadata.yml example:**
```yaml
features:
  velocity:
    name: velocity_x, velocity_y
    group: motion
    requires: [position]
    min_frames: 2
    description: "Storm movement speed (pixels per 10 minutes)"

  delta_size:
    name: delta_size
    group: temporal
    requires: [bbox_area]
    min_frames: 2
    description: "Frame-to-frame change in storm area"
```

---

## 3. Dimensionality Reduction (PCA)

### 3.1 Strategy: Grouped PCA

**Apply PCA to correlated feature groups only:**

```yaml
pca_groups:
  thermal_pca:
    input_features:
      - mean_color
      - max_color
      - min_color
      - std_color
      - color_gradient
    method: elbow_second_derivative
    variance_threshold: 0.95
    selected_pcs: auto  # determined by elbow

  spatial_pca:
    input_features:
      - bbox_width
      - bbox_height
      - bbox_area
      - aspect_ratio
      - x_center
      - y_center
    method: elbow_second_derivative
    variance_threshold: 0.95
    selected_pcs: auto

# NO PCA for these - preserve temporal meaning
raw_features:
  - velocity_x, velocity_y, speed, direction, acceleration
  - delta_size, delta_color_mean, delta_speed
```

### 3.2 Elbow Detection: Second Derivative Method

**Algorithm**:
1. Compute explained variance ratio per PC
2. Calculate first derivative (rate of change)
3. Calculate second derivative (acceleration)
4. Find maximum second derivative (sharpest drop)
5. Select PCs before the elbow point

**Example**:
```
PC1: 16.26% → PC2: 14.45% (drop: -1.81) → PC3: 13.33% (drop: -1.12)
PC2: 14.45% → PC3: 13.33% (drop: -1.12) → PC4: 11.24% (drop: -2.09)
PC3: 13.33% → PC4: 11.24% (drop: -2.09) → PC5: 7.32%  (drop: -3.92) ← ELBOW
PC4: 11.24% → PC5: 7.32%  (drop: -3.92) → PC6: 6.42%  (drop: -0.90)

Second derivative at PC4→PC5: -1.83 (largest magnitude)
→ Select top 4 PCs
```

### 3.3 Feature Standardization
- **Before PCA**: StandardScaler (zero mean, unit variance)
- **Store scalers**: Save for inverse transform during inference
- **Per-group scaling**: Separate scaler per PCA group

### 3.4 PCA Output
**Generated files**:
- `pca_config.yml` - Auto-generated PC selections
- `pca_thermal.pkl` - Fitted PCA transformer
- `pca_spatial.pkl` - Fitted PCA transformer
- `scaler_thermal.pkl` - StandardScaler for thermal features
- `scaler_spatial.pkl` - StandardScaler for spatial features
- `pca_loadings.csv` - Raw feature contributions to each PC

**Loading attribution** (for GUI display):
```csv
feature,PC1_thermal,PC2_thermal,PC3_thermal
mean_color,0.52,0.13,-0.21
max_color,0.48,0.19,0.33
std_color,0.23,0.61,0.45
...
```

---

## 4. Prediction Targets & Horizons

### 4.1 What to Predict (per storm, per horizon)
1. **Position**: `x_center`, `y_center` (normalized coordinates)
2. **Size**: `bbox_area` (normalized)
3. **Intensity**: `mean_color` or `std_color` (thermal proxy)

**Total outputs per storm**: 4 values (x, y, size, intensity)

### 4.2 Time Horizons

**Problem**: 10 minutes too short for meaningful hurricane changes

**Solution**: Multi-horizon forecasting

| Horizon | Timesteps Ahead | Real Time | Use Case |
|---------|----------------|-----------|----------|
| t+1 | 1 frame | 10 minutes | Immediate nowcasting |
| t+3 | 3 frames | 30 minutes | Short-term warning |
| t+6 | 6 frames | 1 hour | Tactical planning |
| t+12 | 12 frames | 2 hours | Strategic forecasting |

**Modeling approach**:
- **Separate model per horizon** (recommended for Phase 1)
- Alternative: Single model with multi-output (Phase 2)

### 4.3 Lookback Window
**Input**: Historical frames `[t-L : t]` where L = lookback length

**Recommended**: L = 6 frames (1 hour of history)
- Captures recent trend
- Not too long (avoid overfitting to noise)
- Configurable in GUI

### 4.4 Multi-Storm Handling

**Strategy**: Storm-level flattened dataset

```
Original: Image with 3 storms → 1 sample with variable outputs
Flattened: Image with 3 storms → 3 independent samples

Each sample = 1 storm + context from same frame
```

**Benefits for tree models**:
- Fixed input/output dimensions
- Simpler training
- Independent predictions per storm

**Requirement**: Storm tracking (to link same storm across frames)

---

## 5. Training Pipeline

### 5.1 Models (Phase 1: Tree-based)

| Model | Use Case | Hyperparameters |
|-------|----------|-----------------|
| **Random Forest** | Baseline, robust | n_estimators, max_depth, min_samples_split |
| **XGBoost** | High performance | learning_rate, max_depth, n_estimators, subsample |
| **LightGBM** | Fast training, handles large features | num_leaves, learning_rate, n_estimators |

**Multi-output approach**: Separate model per target variable
- `model_x`: predicts x_center at t+n
- `model_y`: predicts y_center at t+n
- `model_size`: predicts bbox_area at t+n
- `model_intensity`: predicts mean_color at t+n

**Alternative** (to compare): MultiOutputRegressor wrapper

### 5.2 Train/Test Split Strategy

**Default: Sequence-based Temporal Split**

```python
# Preserve temporal order + respect sequence boundaries
sequences = load_sequences()  # 122 sequences
sequences.sort(by='start_time')

split_ratio = 0.8
split_idx = int(split_ratio * len(sequences))

train_sequences = sequences[:split_idx]  # earlier ~80%
test_sequences = sequences[split_idx:]   # later ~20%
```

**Other options (selectable in GUI)**:
- **Temporal split by date**: Train before cutoff, test after
- **Walk-forward validation**: Rolling window (limited by short sequences)
- **Leave-one-storm-out**: Test generalization (requires storm tracking)

**GUI controls**:
- Split strategy dropdown
- Train ratio slider (60-90%)
- Random seed input
- Preview: Show train/test date ranges and sequence counts

### 5.3 Gradio GUI Features

#### Feature Selection Panel
```
┌─ Raw Features ────────────────────────┐
│ ☑ Spatial Features (6)                │
│   ☑ x_center, y_center               │
│   ☑ bbox_width, bbox_height          │
│   ☑ bbox_area, aspect_ratio          │
│                                        │
│ ☑ Thermal Features (5)                │
│   ☑ mean_color, max_color, ...       │
│                                        │
│ ☑ Motion Features (5)                 │
│   ☑ velocity_x, velocity_y, ...       │
│                                        │
│ ☑ Temporal Features (3)               │
│   ☑ delta_size, delta_color, ...     │
└────────────────────────────────────────┘

┌─ PCA Features ────────────────────────┐
│ ☑ PC1_thermal (16.2% var) ⓘ           │
│   Top: max_color (52%), mean (48%)    │
│ ☑ PC2_thermal (14.5% var) ⓘ           │
│   Top: std_color (61%), gradient (35%)│
│ ☑ PC3_thermal (13.3% var) ⓘ           │
│ ...                                    │
│                                        │
│ ☑ PC1_spatial (28.3% var) ⓘ           │
│ ☑ PC2_spatial (22.1% var) ⓘ           │
└────────────────────────────────────────┘

⚠️ Warning: PC1_thermal includes raw thermal
features. Using both may cause multicollinearity.

[Smart Select] - Auto-resolve conflicts
```

#### Model Configuration
```
Models to Train:
☑ Random Forest
☑ XGBoost
☑ LightGBM
☐ LSTM (requires continuous sequences >10 frames)

Prediction Horizons:
☑ t+1 (10 min)
☑ t+3 (30 min)
☑ t+6 (1 hour)
☑ t+12 (2 hours)

Lookback Window: [6 frames (1 hour)] slider

Train/Test Split:
Strategy: [Sequence-based Temporal ▼]
Train Ratio: [80% ████████░░]
Random Seed: [42]

[Preview Split] → Shows date ranges
```

#### Training Controls
```
[Start Training]  [Stop]  [Export Config]  [Load Config]

Progress:
▓▓▓▓▓▓▓▓░░░░░░░░ 8/12 models (XGBoost t+6)
Elapsed: 2m 34s  |  ETA: 1m 12s
```

---

## 6. Evaluation Metrics

### 6.1 Per-Model, Per-Horizon Metrics

**Output table**:

| Model | Horizon | Train Time (s) | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² |
|-------|---------|----------------|------------|-----------|-----------|----------|----------|---------|
| RF | t+1 | 12.3 | 0.023 | 0.031 | 0.018 | 0.024 | 0.89 | 0.82 |
| XGB | t+1 | 8.7 | 0.019 | 0.028 | 0.015 | 0.021 | 0.92 | 0.85 |
| LGBM | t+1 | 5.2 | 0.021 | 0.029 | 0.016 | 0.022 | 0.91 | 0.84 |
| RF | t+3 | 11.8 | 0.041 | 0.058 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Metrics per target** (x, y, size, intensity):
- RMSE, MAE, R² for each separately
- Aggregated metrics (average across targets)

### 6.2 Visualization Outputs

1. **Predicted vs Actual Trajectories**
   - Plot storm paths: actual (solid) vs predicted (dashed)
   - Color-code by time horizon
   - Show confidence intervals (if applicable)

2. **Error Distribution by Horizon**
   - Box plots: RMSE at t+1, t+3, t+6, t+12
   - Identify which horizon is hardest to predict

3. **Feature Importance**
   - Tree-based importance scores
   - SHAP values (optional, Phase 2)

4. **Residual Analysis**
   - Scatter: predicted vs actual
   - Residual histograms (check for bias)

5. **Per-Sequence Performance**
   - Heatmap: Error per sequence (identify problematic sequences)
   - Metric: Average error per continuous sequence

### 6.3 Physical Constraint Checks

**Post-prediction validation**:
- ❌ Position outside image bounds (x,y ∈ [0,1])
- ❌ Negative size or intensity
- ❌ Unrealistic speed (e.g., >500 km/h for 10-min interval)
- ⚠️ Discontinuous jumps (>3σ from mean velocity)

**Report violations** in evaluation output

### 6.4 Export Results

**Formats**:
- **CSV**: Full predictions + actuals for manual analysis
- **HTML Report**: Interactive dashboard with plots
- **Model files**: Saved `.pkl` for deployment
- **Config YAML**: Reproducible experiment settings

---

## 7. Project Structure

```
GK-2A/
├── CLAUDE.md                          # This file (pipeline specs)
├── dataset_analysis_report.md         # Initial EDA (completed)
├── analyze_dataset.py                 # Initial analysis script
│
├── data/
│   ├── raw/
│   │   ├── Hurricane.v3i.yolov8/      # Original YOLO dataset
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── combined/                  # Merged train+val+test (to create)
│   │
│   ├── processed/
│   │   ├── sequences/                 # Structured sequence data
│   │   │   ├── sequences.pkl
│   │   │   └── sequences_metadata.json
│   │   ├── features/
│   │   │   ├── spatial_features.parquet
│   │   │   ├── thermal_features.parquet
│   │   │   ├── motion_features.parquet
│   │   │   └── temporal_features.parquet
│   │   └── storm_tracking/
│   │       └── storm_tracks.pkl       # Storm ID assignments
│   │
│   └── splits/
│       ├── train_sequences.json
│       └── test_sequences.json
│
├── preprocessing/
│   ├── 1_combine_datasets.py          # Merge train/val/test
│   ├── 2_extract_yolo_features.py     # Parse YOLO bboxes
│   ├── 3_extract_thermal_features.py  # Extract pixel colors
│   ├── 4_build_sequences.py           # Detect gaps, create sequences
│   └── 5_track_storms.py              # Multi-object tracking (TBD via analysis)
│
├── features/
│   ├── metadata.yml                   # Feature definitions
│   ├── core/
│   │   └── sequence_api.py            # Central data access API
│   ├── spatial.py                     # Bbox-derived features
│   ├── thermal.py                     # Color-based features
│   ├── motion.py                      # Velocity, acceleration
│   └── temporal.py                    # Delta features
│
├── pca/
│   ├── pca_analyzer.py                # Grouped PCA with elbow detection
│   ├── config/
│   │   └── pca_config.yml             # Auto-generated (selected PCs)
│   ├── transformers/
│   │   ├── pca_thermal.pkl
│   │   ├── pca_spatial.pkl
│   │   ├── scaler_thermal.pkl
│   │   └── scaler_spatial.pkl
│   └── analysis/
│       └── pca_loadings.csv           # Feature attributions
│
├── training/
│   ├── train_gui.py                   # Gradio interface (main entry)
│   ├── models/
│   │   ├── tree_models.py             # RF, XGB, LGBM implementations
│   │   └── lstm_model.py              # Phase 2
│   ├── splits/
│   │   └── split_strategies.py        # Temporal, sequence-based, etc.
│   ├── configs/
│   │   ├── default_config.yml
│   │   └── experiments/               # Saved experiment configs
│   └── trained_models/
│       ├── rf_t1_x.pkl
│       ├── xgb_t3_y.pkl
│       └── ...
│
├── evaluation/
│   ├── metrics.py                     # RMSE, MAE, R² calculations
│   ├── visualize_trajectories.py     # Plot predicted paths
│   ├── visualize_errors.py           # Error distributions
│   ├── feature_importance.py         # Tree importances
│   └── generate_report.py            # HTML report generator
│
├── analysis/
│   ├── combined_dataset_analysis.py   # Analysis of merged dataset (subagent task)
│   ├── storm_tracking_analysis.py    # Derive tracking function (subagent task)
│   └── reports/
│       ├── combined_data_report.md
│       └── storm_tracking_report.md
│
├── utils/
│   ├── logger.py                      # Logging utilities
│   ├── config_loader.py               # YAML config parser
│   └── cache_manager.py               # Feature caching
│
├── cache/                              # Temporary cached data
│   ├── features/
│   ├── sequences/
│   └── models/
│
└── requirements.txt
```

---

## 8. Implementation Phases

### Phase 1: Foundation (Current)

**Step 1**: Project structure setup
- Create directory structure
- Setup requirements.txt
- Initialize logging

**Step 2**: Data analysis (parallel subagent tasks)
- **Subagent A**: Analyze combined train+val+test dataset
  - Temporal continuity of merged data
  - Storm count distributions
  - Identify optimal continuous sequences

- **Subagent B**: Storm tracking analysis
  - Analyze spatial proximity patterns
  - Test similarity metrics (position, size, thermal)
  - Propose tracking algorithm

**Step 3**: Preprocessing pipeline
- Combine datasets
- Extract YOLO features
- Extract thermal features (pixel colors)
- Build sequences
- Implement storm tracking (based on subagent findings)

**Step 4**: Feature engineering
- Central sequence API
- Implement feature modules
- Generate feature metadata

**Step 5**: PCA analysis
- Implement grouped PCA
- Second derivative elbow detection
- Generate PCA config and loadings

**Step 6**: Training GUI
- Gradio interface
- Model training logic
- Split strategies

**Step 7**: Evaluation
- Metrics calculation
- Visualization
- Report generation

**Step 8**: Interactive Visualization GUI ✅ **COMPLETED**
- Gradio + Plotly interface for zoomable satellite imagery
- Real-time prediction overlay on test set
- Error metrics display and analysis
- Public sharing capability (ngrok/Gradio share)

### Phase 2: Deep Learning (Future)
- LSTM implementation
- Transformer experiments (if data grows)
- Ensemble methods

---

## 9. Key Technical Decisions

| Decision Point | Choice | Rationale |
|----------------|--------|-----------|
| **PCA Scope** | Grouped (thermal + spatial only) | Preserve temporal feature interpretability |
| **PCA Selection** | Second derivative elbow | Automatic, robust falloff detection |
| **Feature Scaling** | StandardScaler per group | Necessary for PCA, preserve group context |
| **Prediction Targets** | (x, y, size, intensity) × 4 horizons | Comprehensive storm forecast |
| **Time Horizons** | t+1, t+3, t+6, t+12 | Balance granularity & physical meaning |
| **Lookback Window** | 6 frames (1 hour) | Capture trend without overfitting |
| **Multi-Storm** | Flattened storm-level dataset | Simpler for tree models, fixed dimensions |
| **Model Architecture** | Separate model per target | Independent, easier debugging |
| **Train/Test Split** | Sequence-based temporal | Respect time order & gaps, avoid leakage |
| **Storm Tracking** | TBD via data analysis | Data-driven approach |
| **Thermal Extraction** | Pixel color values | Direct measurement from imagery |
| **Primary Models** | RF, XGBoost, LightGBM | Pragmatic for dataset size, fast iteration |
| **Implementation** | Bottom-up | Build strong foundation first |

---

## 10. Open Questions (Pending Analysis)

1. **Storm Tracking Algorithm**
   - Spatial proximity threshold?
   - Weight for size/thermal similarity?
   - Hungarian algorithm vs greedy matching?
   - **Action**: Subagent analysis in progress

2. **Combined Dataset Characteristics**
   - How many total sequences?
   - Longest continuous sequence?
   - Storm appearance patterns?
   - **Action**: Subagent analysis in progress

3. **Thermal Feature Details**
   - RGB vs grayscale pixel values?
   - Normalization strategy?
   - Spatial patterns within bbox (future: gradient maps)?

4. **Hyperparameter Tuning**
   - Grid search vs Bayesian optimization?
   - Cross-validation strategy (given sequence structure)?

---

## 11. Success Criteria

### Minimum Viable Product (MVP)
- ✅ Complete preprocessing pipeline
- ✅ All feature groups implemented
- ✅ PCA working with auto-selection
- ✅ Training GUI functional
- ✅ 3 tree models trained on at least t+1, t+3
- ✅ Evaluation metrics + basic visualizations
- ✅ Interactive forecast visualization GUI with zoom capability

### Performance Targets (Phase 1) ✅ **ACHIEVED**
- ✅ **t+1 (10 min)**: Test R² = 0.862 (LightGBM Stacking) > 0.75 target
- ✅ **t+3 (30 min)**: Test R² = 0.817 (LightGBM Stacking) > 0.60 target
- ✅ **t+6 (1 hour)**: Test R² = 0.761 (LightGBM Stacking) > 0.45 target
- ✅ **t+12 (2 hours)**: Test R² = 0.595 (LightGBM baseline)
- ✅ **Training time**: <5 minutes per model on laptop

### Phase 2 Targets
- LSTM outperforms tree models on t+6, t+12
- Real-time inference (<100ms per prediction)
- Deployable API

---

---

## 12. Visualization GUI

### Interactive Forecast Visualization

**Location**: `visualization/gradio_app.py`

**Features**:
- **Zoomable satellite imagery** using Plotly for interactive pan/zoom
- **Real-time prediction overlay** on test set (642 frames, Oct 17-21, 2023)
- **Multi-horizon forecasts**: t+1 (10min), t+3 (30min), t+6 (1hr), t+12 (2hrs)
- **Color-coded trajectories**:
  - Magenta boxes: Current storm positions (YOLO detections)
  - White lines: Actual future paths (ground truth)
  - Purple dots: t+1 predictions (86% confidence)
  - Bright green dots: t+3 predictions (82% confidence)
  - Pink dots: t+6 predictions (76% confidence)
  - Cyan dots: t+12 predictions (59% confidence)
- **Error metrics**: Position offset % (1% ≈ 75-100 km on full Earth disk)
- **Interactive controls**: Timeline slider, prev/next navigation, toggle predictions

**Running the GUI**:
```bash
python visualization/gradio_app.py
```

**Public sharing**:
- **Option 1**: Gradio share (built-in, 72-hour temporary link)
  - Already enabled with `share=True` in code
  - Public URL appears in terminal output

- **Option 2**: ngrok (persistent with account)
  ```bash
  # Install
  brew install ngrok/ngrok/ngrok

  # Get token from https://dashboard.ngrok.com/get-started/your-authtoken
  ngrok config add-authtoken YOUR_TOKEN

  # Start app
  python visualization/gradio_app.py

  # In new terminal
  ngrok http 7862
  ```

**Data Structure**:
- Test set: 2101 samples, 642 frames, 3 sequences
- Date range: Oct 17-21, 2023
- 100% image coverage (642/642 satellite images found)

---

*Last Updated: 2025-11-19*
*Status: Phase 1 - **COMPLETE** ✅*
