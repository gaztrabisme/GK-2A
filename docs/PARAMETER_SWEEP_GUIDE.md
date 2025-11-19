# Parameter Sweep / AutoML Guide

## Overview

The Parameter Sweep feature (Tab 5) implements automated hyperparameter search using randomized search to find optimal model configurations.

## What It Does

The sweep engine automatically tests different combinations of:
- **Split strategies**: sequence, date, random
- **Train/test ratios**: Randomized within ±10% of base value
- **Feature sets**: pca_only, raw_only, motion_temporal_only, random_subset, all_features
- **Prediction targets**: x_center, y_center, bbox_area, mean_color
- **Time horizons**: t+1, t+3, t+6, t+12
- **Models**: random_forest, xgboost, lightgbm

## How to Use

### Step 1: Load Data
1. Go to Tab 1: "Data & Features"
2. Click "Load Data"
3. Wait for data to load successfully

### Step 2: Configure Sweep
Go to Tab 5: "Parameter Sweep" and configure:

**Number of Iterations** (10-500)
- Start with 50-100 for initial exploration
- Use 200-500 for thorough search

**Split Strategies**
- `sequence`: Recommended - prevents data leakage
- `date`: May have leakage if sequences span split boundary
- `random`: Breaks temporal order, not recommended for time-series

**Train Ratio Range**
- Base value (0.6-0.9), sweep will randomize ±10%
- Default: 0.8 (80% train, 20% test)

**Feature Strategies**
- `pca_only`: Use PCA components + motion + temporal
- `raw_only`: Use raw features (no PCA)
- `motion_temporal_only`: Only motion and temporal features (no spatial/thermal)
- `random_subset`: Randomly select 5-N features
- `all_features`: Use all available features

**Prediction Targets**
- `x_center`, `y_center`: Storm position
- `bbox_area`: Storm size
- `mean_color`: Storm intensity (thermal proxy)

**Time Horizons**
- `1`: 10 minutes ahead
- `3`: 30 minutes ahead
- `6`: 1 hour ahead
- `12`: 2 hours ahead

**Models to Try**
- `random_forest`: Robust baseline
- `xgboost`: Often best performance
- `lightgbm`: Fast training

### Step 3: Run Sweep
1. Click "🚀 Start Sweep"
2. Monitor live progress in the log
3. Watch best R² update in real-time
4. View top 10 configurations in leaderboard

### Step 4: Analyze Results
After sweep completes:

**Leaderboard Table**
- Shows top 10 configurations by Test R²
- Columns: iteration, model, horizon, target, n_features, test_r2, test_rmse, train_r2, split_strategy

**Best Configuration Details**
- JSON display of the best performing configuration
- Includes all hyperparameters

**Export Results**
- Click "💾 Export Results" to save full results to CSV
- File saved as `training/sweep_results.csv`
- Contains ALL iterations with full details

## Interpreting Results

### Good vs Bad R² Values

**Position Prediction (x_center, y_center)**
- t+1: R² > 0.65 = Good, R² > 0.75 = Excellent
- t+3: R² > 0.50 = Good, R² > 0.60 = Excellent
- t+6: R² > 0.40 = Good, R² > 0.55 = Excellent
- t+12: R² > 0.25 = Good, R² > 0.40 = Excellent

**Size Prediction (bbox_area)**
- Generally harder to predict
- R² > 0.30 is reasonable

**Intensity Prediction (mean_color)**
- Depends on thermal variability
- R² > 0.40 is good

### Warning Signs

**Perfect R² (>0.95)**
- Likely data leakage!
- Check if using date-based split with sequence overlap
- Check if spatial PCA features included when predicting position

**Negative R²**
- Model worse than predicting the mean
- Feature set may not contain useful information
- Example: Predicting position with only motion features (no current position)

**No Horizon Degradation**
- If t+1, t+3, t+6 all have same R², something is wrong
- Should see decreasing R² with longer horizons

## Feature Strategy Recommendations

### For Position Prediction
**Best**: `pca_only` or `raw_only`
- Need current position to predict future position
- Spatial features are essential

**Avoid**: `motion_temporal_only`
- Cannot predict absolute position from motion alone
- Results in negative R²

### For Size/Intensity Prediction
**Try**: `motion_temporal_only` or `random_subset`
- May work without spatial features
- Less risk of target leakage

## Common Issues

### "No valid pairs found for horizon"
- Dataset too short for the horizon
- Solution: Use shorter horizons or longer sequences

### "Insufficient test samples"
- Split created very small test set
- Solution: Adjust train ratio or use different split strategy

### Very slow progress
- Large number of iterations with complex models
- Solution: Start with fewer iterations (20-50) or use only random_forest

## Best Practices

1. **Start Small**: Run 20-30 iterations first to validate setup
2. **Use Sequence Split**: Prevents data leakage in time-series data
3. **Check Leaderboard**: Look for consistent patterns in top configs
4. **Export Results**: Save all results for later analysis
5. **Multiple Runs**: Run sweep multiple times to verify stability

## Example Workflow

```
1. Load data (Tab 1)
2. Run quick manual test (Tabs 2-3) to verify data is working
3. Go to Tab 5 - Parameter Sweep
4. Configure:
   - 50 iterations
   - sequence split only
   - pca_only, raw_only, motion_temporal_only
   - x_center, y_center
   - horizons 1, 3, 6
   - random_forest, xgboost
5. Start sweep
6. While running: watch for errors in log
7. After completion:
   - Review leaderboard
   - Identify best configuration
   - Export results
8. Re-run best config manually (Tabs 2-3) to verify
9. Save best model for deployment
```

## Output Files

**sweep_results.csv**
- Full results from all iterations
- Columns include:
  - iteration, model, horizon, target, split_strategy, train_ratio
  - n_features, features (list), test_r2, test_rmse, train_r2, train_rmse
  - train_time, elapsed_time
  - error (if iteration failed)

**Analysis in spreadsheet**
- Sort by test_r2 descending
- Filter by specific target or horizon
- Plot R² vs n_features to find sweet spot
- Compare split strategies

---

*Created: 2025-11-18*
*Part of Hurricane Storm Forecasting Pipeline*
