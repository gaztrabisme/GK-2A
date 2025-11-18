# Multi-Horizon Training Fix

## Problem
Previously, selecting multiple time horizons (t+1, t+3, t+6, t+12) had no effect on training. All horizons produced identical R² scores because the code was only training once on the current frame data.

## Solution
Implemented proper time-series forecasting with horizon-specific datasets.

### Key Changes

1. **Created `horizon_data_builder.py`**
   - Builds time-shifted datasets for each horizon
   - For horizon t+6: pairs features_at_frame_t with target_at_frame_(t+6)
   - Only includes valid pairs where both current and future frames exist in same sequence
   - Respects sequence boundaries (won't cross sequence gaps)

2. **Updated `train_gui.py`**
   - Training loop now iterates over each selected horizon
   - Each horizon gets its own dataset with proper time shifts
   - Models are named with horizon suffix: `random_forest_t1`, `random_forest_t3`, etc.
   - Results table shows separate metrics per horizon

### How It Works

**Example: Training on t+1 and t+6**

**t+1 Dataset (10 minutes ahead):**
```
Current Frame (t) → Target Frame (t+1)
Frame 0 → Frame 1
Frame 1 → Frame 2
Frame 2 → Frame 3
...
Frame 99 → Frame 100
```
Result: ~2,300 training pairs

**t+6 Dataset (1 hour ahead):**
```
Current Frame (t) → Target Frame (t+6)
Frame 0 → Frame 6
Frame 1 → Frame 7
Frame 2 → Frame 8
...
Frame 94 → Frame 100
```
Result: ~2,100 training pairs (fewer because we need 6 frames ahead)

### Expected Behavior

**Before Fix:**
- t+1: R² = 0.717
- t+3: R² = 0.717 (identical - wrong!)
- t+6: R² = 0.717 (identical - wrong!)

**After Fix:**
- t+1: R² = 0.717 (baseline)
- t+3: R² = 0.55-0.65 (harder to predict 30 min ahead)
- t+6: R² = 0.40-0.55 (even harder for 1 hour ahead)
- t+12: R² = 0.20-0.40 (very challenging for 2 hours)

This degradation is **expected and correct** - predicting further into the future is inherently harder!

### Validation

To verify it's working:
1. Train with t+1, t+3, t+6 selected
2. Check training log - should see different dataset sizes:
   - t+1: ~2,316 samples
   - t+3: ~2,222 samples
   - t+6: ~2,128 samples
3. Check results - R² should decrease with horizon
4. If R² increases, something is wrong (re-check)

### Technical Details

**HorizonDataBuilder.build_horizon_dataset():**
- Input: Full dataframe, horizon value
- Output: (X_current, y_future, indices)
- Process:
  1. Group by track_id (each storm's time series)
  2. For each frame, look for frame+horizon in same track
  3. Verify both frames in same sequence (no gap crossing)
  4. Create feature-target pair
  5. Apply train/test split by sequences

**Key Constraint:**
Only creates pairs where future frame exists. For t+12, we lose the last 12 frames of each sequence, reducing dataset size.

### Files Modified
- `training/horizon_data_builder.py` (new)
- `training/train_gui.py` (updated run_training function)
- Tab 2: Added t+12 option

### Next Steps
- Test with all 4 horizons
- Verify R² degradation curve
- Use for parameter sweep experiments
