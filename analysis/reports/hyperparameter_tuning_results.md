# Hyperparameter Tuning Results - First Attempt

**Date**: 2025-11-19  
**Status**: ❌ Failed - Invalid validation split

## Problem

The 60/20/20 sequence-based split resulted in:
- Train: 321 samples (4 sequences)
- **Val: 4 samples (1 sequence)** ← Only 4 validation samples!
- Test: 1749 samples (3 sequences)

With only 4 validation samples, Optuna's optimization was unreliable.

## Results Comparison

### Before Tuning (Baseline with Default Hyperparameters)

| Model | t+1 R² | t+3 R² | t+6 R² |
|-------|--------|--------|--------|
| Random Forest | 0.6943 | 0.6515 | 0.6268 |
| XGBoost | 0.6592 | 0.6856 | 0.6818 |
| LightGBM | 0.7238 | 0.6727 | 0.5762 |
| **Ensemble** | **0.7137** | **0.6895** | **0.6589** |

### After Tuning (With Optimized Hyperparameters)

| Model | t+1 R² | t+3 R² | t+6 R² | Change |
|-------|--------|--------|--------|--------|
| Random Forest | 0.4878 | 0.4382 | 0.4158 | **-0.21** ❌ |
| XGBoost | 0.4455 | 0.4241 | 0.4206 | **-0.24** ❌ |
| LightGBM | 0.8017 | 0.7267 | 0.6659 | **+0.08** ✅ |
| **Ensemble** | **0.6205** | **0.5652** | **0.5327** | **-0.12** ❌ |

## Key Findings

1. **LightGBM improved significantly**: +0.08-0.09 R² across all horizons
2. **Random Forest degraded**: -0.21 R² (over-regularized)
3. **XGBoost degraded**: -0.24 R² (over-regularized)
4. **Ensemble got worse**: -0.12 R² (because 2/3 models degraded)

## Tuned Hyperparameters

### LightGBM (Successful)
- n_estimators: 326
- num_leaves: 85
- learning_rate: 0.0801
- feature_fraction: 0.9567
- bagging_fraction: 0.6789
- min_child_samples: 41
- reg_alpha: 0.1265
- reg_lambda: 7.0661 ← Strong L2 regularization helped!

### Random Forest (Over-regularized)
- n_estimators: 103
- max_depth: 25
- min_samples_split: 39 ← Too high
- min_samples_leaf: 16 ← Too high
- max_features: 0.5

### XGBoost (Over-regularized)
- n_estimators: 133
- max_depth: 7
- learning_rate: 0.0101 ← Too low
- subsample: 0.9550
- colsample_bytree: 0.7338
- reg_alpha: 0.8549
- reg_lambda: 8.4392 ← Too high

## Conclusion

The tuning infrastructure works, but the validation split must be fixed. With only 4 validation samples, Optuna over-regularized RF and XGBoost.

## Next Steps

1. Fix validation split to ensure adequate samples (>100)
2. Consider sample-based split or k-fold CV within sequences
3. Re-run tuning with proper validation set
