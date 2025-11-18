# YOLO Training Dataset Analysis Report

## Dataset Overview

- **Total Images**: 576
- **Time Range**: 2023-10-15 21:10 to 2023-10-21 14:10
- **Total Duration**: 5 days, 17:00:00
- **Image Format**: 678x678 pixels, thermal satellite imagery (GOES-18)
- **Class**: Hurricane (Class 0)

---

## 1. Temporal Continuity Analysis

### Gap Statistics

The expected time interval between consecutive images is **10 minutes**.

| Statistic | Value (minutes) |
|-----------|----------------|
| Mean | 14.30 |
| Median | 10.00 |
| Min | 10.00 |
| Max | 60.00 |
| Std Dev | 8.18 |

### Gap Distribution

| Gap Duration | Count | Percentage |
|--------------|-------|------------|
| 10min (expected) | 410 | 71.3% |
| 10-30min | 145 | 25.2% |
| 30-60min | 20 | 3.5% |
| 1-6h | 0 | 0.0% |
| 6-24h | 0 | 0.0% |
| >24h | 0 | 0.0% |

### Data Continuity Summary

- **Continuous sequences**: 122 sequences with ≤10 minute gaps
- **Expected 10-minute intervals**: 410 out of 575 transitions (71.3%)
- **Data gaps**: 165 gaps detected

### Top 10 Longest Continuous Sequences

| Rank | Frames | Start Time | End Time | Duration |
|------|--------|------------|----------|----------|
| 1 | 15 | 2023-10-17 05:10 | 2023-10-17 07:30 | 2:20:00 |
| 2 | 10 | 2023-10-16 08:10 | 2023-10-16 09:40 | 1:30:00 |
| 3 | 10 | 2023-10-18 19:30 | 2023-10-18 21:00 | 1:30:00 |
| 4 | 10 | 2023-10-19 03:20 | 2023-10-19 04:50 | 1:30:00 |
| 5 | 10 | 2023-10-20 00:00 | 2023-10-20 01:30 | 1:30:00 |
| 6 | 10 | 2023-10-20 06:10 | 2023-10-20 07:40 | 1:30:00 |
| 7 | 9 | 2023-10-16 12:20 | 2023-10-16 13:40 | 1:20:00 |
| 8 | 9 | 2023-10-19 01:30 | 2023-10-19 02:50 | 1:20:00 |
| 9 | 9 | 2023-10-20 01:50 | 2023-10-20 03:10 | 1:20:00 |
| 10 | 8 | 2023-10-16 15:00 | 2023-10-16 16:10 | 1:10:00 |

---

## 2. Bounding Box Analysis

### Overall Statistics

- **Images with bounding boxes**: 573
- **Images without bounding boxes**: 3
- **Total bounding boxes**: 1798

### Bounding Boxes per Image

| Statistic | Value |
|-----------|-------|
| Mean | 3.14 |
| Median | 3.00 |
| Min | 1 |
| Max | 7 |
| Std Dev | 1.18 |

### Bounding Box Size Statistics

All values are in normalized coordinates [0-1].

#### Width
| Statistic | Value |
|-----------|-------|
| Mean | 0.0699 |
| Median | 0.0672 |
| Min | 0.0133 |
| Max | 0.2453 |
| Std Dev | 0.0282 |

#### Height
| Statistic | Value |
|-----------|-------|
| Mean | 0.0899 |
| Median | 0.0820 |
| Min | 0.0187 |
| Max | 0.2359 |
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
| Mean | 0.8499 |
| Median | 0.7626 |
| Min | 0.2721 |
| Max | 8.6667 |
| Std Dev | 0.4864 |

---

## 3. Key Insights

### Temporal Characteristics
1. The dataset contains significant temporal gaps, with only 71.3% of transitions at the expected 10-minute interval
2. The longest gap is 60.00 minutes (1.0 hours)
3. There are 122 continuous sequences suitable for time-series analysis
4. The longest continuous sequence contains 15 frames spanning 2:20:00

### Hurricane Detection Characteristics
1. On average, each image contains 3.14 hurricane detections
2. Hurricane bounding boxes have a mean area of 0.0066 (normalized)
3. The aspect ratio averages 0.8499, indicating mostly square hurricane patterns
4. 3 images contain no hurricane detections

---

## 4. Recommendations for Time-Series Modeling

1. **Sequence Selection**: Use the identified continuous sequences for training time-series models to avoid temporal discontinuities
2. **Gap Handling**: Consider imputation or special handling for gaps larger than 10 minutes
3. **Multi-Target Tracking**: With an average of 3.14 hurricanes per image, implement multi-object tracking
4. **Size Variation**: The standard deviation of 0.0045 in bbox area indicates significant size variation - consider scale-aware features
5. **Spatial Coverage**: Analyze the spatial distribution of hurricane centers to understand typical storm trajectories

---

*Report generated on 2025-11-17 16:03:58*
