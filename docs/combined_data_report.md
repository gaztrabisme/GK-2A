# Combined Dataset Analysis Report
## Hurricane Forecasting Time-Series Dataset

---

## Executive Summary

This report analyzes the **combined train+valid+test** dataset to evaluate temporal continuity for time-series hurricane forecasting. The dataset contains **822 total images** across three splits, spanning from **2023-10-15 21:00** to **2023-10-21 14:20**.

**Key Findings:**
- Combined dataset shows **818 continuous 10-minute intervals** (99.6% of transitions)
- Identified **4 continuous sequences** suitable for time-series modeling
- Longest continuous sequence: **573 frames** spanning **3 days, 23:20:00**
- Total of **2577 hurricane bounding boxes** detected across all images

---

## 1. Dataset Overview

### Split Distribution

| Split | Image Count | Percentage | Bounding Boxes |
|-------|-------------|------------|----------------|
| Train | 576 | 70.1% | 1798 |
| Valid | 164 | 20.0% | 527 |
| Test | 82 | 10.0% | 252 |
| **Total** | **822** | **100.0%** | **2577** |

### Dataset Characteristics

- **Total Images**: 822
- **Time Range**: 2023-10-15 21:00 to 2023-10-21 14:20
- **Total Duration**: 5 days, 17:20:00
- **Image Format**: 678x678 pixels, thermal satellite imagery (GEO-KOMPSAT-2A)
- **Class**: Storm/Hurricane (Class 0)
- **Expected Temporal Resolution**: 10 minutes

---

## 2. Temporal Continuity Analysis

### Gap Statistics

The expected time interval between consecutive images is **10 minutes**.

| Statistic | Value (minutes) |
|-----------|----------------|
| Mean | 10.04 |
| Median | 10.00 |
| Min | 10.00 |
| Max | 20.00 |
| Std Dev | 0.60 |

### Gap Distribution

| Gap Duration | Count | Percentage |
|--------------|-------|------------|
| 10min (expected) | 818 | 99.6% |
| 10-30min | 3 | 0.4% |
| 30-60min | 0 | 0.0% |
| 1-6h | 0 | 0.0% |
| 6-24h | 0 | 0.0% |
| >24h | 0 | 0.0% |

### Data Continuity Summary

- **Continuous sequences**: 4 sequences with ≤10 minute gaps
- **Expected 10-minute intervals**: 818 out of 821 transitions (99.6%)
- **Data gaps**: 3 gaps detected
- **Average gap**: 10.04 minutes
- **Longest gap**: 20.00 minutes (0.3 hours)

### Top 15 Longest Continuous Sequences

| Rank | Frames | Start Time | End Time | Duration | Split Distribution |
|------|--------|------------|----------|----------|--------------------|
| 1 | 573 | 2023-10-17 15:00 | 2023-10-21 14:20 | 3 days, 23:20:00 | Test: 56, Train: 399, Valid: 118 |
| 2 | 134 | 2023-10-16 04:40 | 2023-10-17 02:50 | 22:10:00 | Test: 9, Train: 98, Valid: 27 |
| 3 | 70 | 2023-10-17 03:10 | 2023-10-17 14:40 | 11:30:00 | Test: 10, Train: 49, Valid: 11 |
| 4 | 45 | 2023-10-15 21:00 | 2023-10-16 04:20 | 7:20:00 | Test: 7, Train: 30, Valid: 8 |

### Sequence Length Distribution

| Sequence Length | Count | Total Frames |
|----------------|-------|--------------|
| 2-5 frames | 0 | 0 |
| 6-10 frames | 0 | 0 |
| 11-20 frames | 0 | 0 |
| 21-50 frames | 1 | 45 |
| 51-100 frames | 1 | 70 |
| 100+ frames | 2 | 707 |

---

## 3. Bounding Box Analysis

### Overall Statistics

- **Images with bounding boxes**: 818 (99.5%)
- **Images without bounding boxes**: 4 (0.5%)
- **Total bounding boxes**: 2577

### Bounding Boxes per Image

| Statistic | Value |
|-----------|-------|
| Mean | 3.15 |
| Median | 3.00 |
| Min | 1 |
| Max | 8 |
| Std Dev | 1.20 |

### Bounding Box Size Statistics

All values are in normalized coordinates [0-1].

#### Width
| Statistic | Value |
|-----------|-------|
| Mean | 0.0700 |
| Median | 0.0680 |
| Min | 0.0133 |
| Max | 0.2453 |
| Std Dev | 0.0281 |

#### Height
| Statistic | Value |
|-----------|-------|
| Mean | 0.0899 |
| Median | 0.0820 |
| Min | 0.0187 |
| Max | 0.2578 |
| Std Dev | 0.0351 |

#### Area
| Statistic | Value |
|-----------|-------|
| Mean | 0.0066 |
| Median | 0.0053 |
| Min | 0.0005 |
| Max | 0.0323 |
| Std Dev | 0.0045 |

#### Aspect Ratio (width/height)
| Statistic | Value |
|-----------|-------|
| Mean | 0.8513 |
| Median | 0.7703 |
| Min | 0.1438 |
| Max | 8.6667 |
| Std Dev | 0.4804 |

---

## 4. Comparison: Combined vs Train-Only Analysis

### Train-Only Dataset (Previous Analysis)
- **Total Images**: 576
- **Time Range**: 2023-10-15 21:10 to 2023-10-21 14:10 (5 days, 17:00:00)
- **Continuous Sequences**: 122 sequences
- **Longest Sequence**: 15 frames (2:20:00)
- **10-min Intervals**: 71.3%
- **Total Bounding Boxes**: 1,798

### Combined Dataset (Current Analysis)
- **Total Images**: 822 (+246 images, +42.7%)
- **Time Range**: 2023-10-15 21:00 to 2023-10-21 14:20 (5 days, 17:20:00)
- **Continuous Sequences**: 4 sequences (-118 sequences, -96.7%)
- **Longest Sequence**: 573 frames (3 days, 23:20:00)
- **10-min Intervals**: 99.6% (28.3% difference)
- **Total Bounding Boxes**: 2577 (+779, +43.3%)

### Key Improvements from Combining Datasets

2. **Longer Maximum Sequence**: 573 frames vs 15 frames (3720.0% increase)
3. **Extended Time Coverage**: 5 days, 17:20:00 total duration
4. **More Training Data**: 2577 total bounding boxes for feature extraction

---

## 5. Optimal Sequences for Time-Series Modeling

Based on the analysis, the following sequences are recommended for time-series forecasting:

### Criteria for Selection
1. **Minimum length**: ≥10 frames (providing sufficient temporal context)
2. **Continuity**: No gaps >10 minutes
3. **Cross-split sequences**: Sequences spanning multiple splits for better generalization

### Top Recommended Sequences (≥20 frames)

| Sequence | Frames | Start | End | Duration | Splits |
|----------|--------|-------|-----|----------|--------|
| 1 | 573 | 2023-10-17 15:00 | 2023-10-21 14:20 | 3 days, 23:20:00 | Test: 56, Train: 399, Valid: 118 |
| 2 | 134 | 2023-10-16 04:40 | 2023-10-17 02:50 | 22:10:00 | Test: 9, Train: 98, Valid: 27 |
| 3 | 70 | 2023-10-17 03:10 | 2023-10-17 14:40 | 11:30:00 | Test: 10, Train: 49, Valid: 11 |
| 4 | 45 | 2023-10-15 21:00 | 2023-10-16 04:20 | 7:20:00 | Test: 7, Train: 30, Valid: 8 |


### Medium-Length Sequences (10-19 frames)


---

## 6. Key Insights and Recommendations

### Temporal Characteristics
1. **Dataset Continuity**: 99.6% of transitions maintain the expected 10-minute interval
2. **Temporal Coverage**: Dataset spans 5 days, 17:20:00, providing good diversity
3. **Sequence Distribution**: 4 continuous sequences suitable for time-series modeling
4. **Longest Sequence**: 573 frames (3 days, 23:20:00), sufficient for LSTM/GRU models

### Hurricane Detection Characteristics
1. **Detection Coverage**: 99.5% of images contain hurricane detections
2. **Multi-Object Tracking**: Average of 3.15 storms per image requires multi-object tracking
3. **Storm Size**: Mean bounding box area of 0.0066 (normalized)
4. **Shape Characteristics**: Aspect ratio of 0.8513 indicates mostly square storm patterns

### Recommendations for Time-Series Forecasting Pipeline

1. **Data Preparation**
   - Use combined dataset and re-split based on temporal sequences
   - Ensure train/val/test splits maintain temporal continuity within sequences
   - Consider using sequences ≥10 frames for training

2. **Feature Engineering**
   - Extract YOLO features: `x_center`, `y_center`, `bbox_width`, `bbox_height`, `bbox_area`, `aspect_ratio`
   - Derive motion features: speed, direction, acceleration
   - Extract thermal features from bounding box regions (mean/max/min temperature, gradients)
   - Compute temporal deltas: Δsize, Δtemperature, Δspeed

3. **Model Architecture**
   - Use LSTM/GRU with attention mechanism for temporal modeling
   - Input sequence length: 5-10 frames (50-100 minutes of history)
   - Prediction horizon: 1-6 frames ahead (10-60 minutes)
   - Handle variable number of storms per frame with object tracking

4. **Training Strategy**
   - Train on continuous sequences to avoid temporal discontinuities
   - Use data augmentation carefully (preserve temporal consistency)
   - Implement multi-object tracking to handle 3.15 avg storms/image
   - Validate on held-out temporal sequences, not random samples

5. **Evaluation Metrics**
   - Track position error (km or pixels)
   - Size prediction error (area, aspect ratio)
   - Multi-step prediction accuracy
   - Storm trajectory tracking over time

---

## 7. Dataset Quality Assessment

### Strengths
- Large dataset with 822 images and 2577 storm annotations
- 4 sequences with ≥10 frames for time-series modeling
- Good temporal resolution (10-minute intervals for 99.6% of data)
- Multi-storm scenarios (3.15 avg storms/image) for complex tracking

### Limitations
- 3 temporal gaps requiring special handling
- Longest gap: 20.00 minutes (0.3 hours)
- 4 images without storm detections
- Original train/valid/test splits may break temporal continuity

### Mitigation Strategies
1. **Gap Handling**: Exclude sequences with gaps >10 minutes or use interpolation
2. **Re-splitting**: Create new train/val/test splits that respect temporal boundaries
3. **Sequence Filtering**: Focus on 4 continuous sequences
4. **Missing Data**: Use forward-filling or temporal interpolation for short gaps

---

## 8. Next Steps

1. **Temporal Re-splitting**
   - Identify 4 suitable sequences
   - Split into train/val/test while maintaining temporal continuity
   - Recommended ratio: 70% train, 15% validation, 15% test

2. **Feature Extraction Pipeline**
   - Extract YOLO-based features from all 2577 bounding boxes
   - Compute temporal deltas between consecutive frames
   - Extract thermal statistics from satellite imagery

3. **Multi-Object Tracking**
   - Implement tracking algorithm to link storms across frames
   - Handle 8 max storms per image
   - Use Hungarian algorithm or DeepSORT for association

4. **Time-Series Model Development**
   - Experiment with LSTM, GRU, and Transformer architectures
   - Implement attention mechanisms for long-range dependencies
   - Test multi-step prediction (1-6 frames ahead)

---

*Report generated on 2025-11-18 08:34:30*
*Analysis script: analyze_combined_dataset.py*
*Visualization: analysis/reports/combined_dataset_analysis.png*
