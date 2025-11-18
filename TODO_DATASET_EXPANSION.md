# Dataset Expansion TODO

## 🎯 Goal
Expand hurricane dataset from 822 images (2023) to 5,000-100,000+ images (2024-2025) using auto-annotation.

---

## 📊 Current Status

### Existing Dataset
- **Period**: October 15-21, 2023 (5 days)
- **Images**: 822 annotated images
- **Storm tracks**: 94 tracks (longest: 289 frames)
- **Annotations**: 2,577 hurricane bounding boxes
- **Quality**: Manually annotated (high quality)

### Available Data Source
- **URL Pattern**: `https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/Sandwich/YYYYDDDHHM0_GOES18-ABI-FD-Sandwich-678x678.jpg`
- **Time Range**: 2023 - present (2025+)
- **Temporal Resolution**: 10-minute intervals
- **Format**: 678x678 pixel thermal satellite imagery

---

## 🛠️ Implementation Plan

### Phase 1: Validation & YOLO Training (5 hours)

#### Step 1.1: Download Test Dataset
**Script**: `preprocessing/6_download_noaa_data.py`

```python
def download_noaa_images(start_date, end_date, output_dir):
    """
    Download GOES-18 images from NOAA CDN

    URL Pattern: https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/Sandwich/
                 YYYYDDDHHM0_GOES18-ABI-FD-Sandwich-678x678.jpg

    Args:
        start_date: datetime (e.g., 2024-09-01)
        end_date: datetime (e.g., 2024-09-15)
        output_dir: Where to save images
    """
    # Implementation:
    # 1. Generate all 10-minute intervals between start_date and end_date
    # 2. Format as YYYYDDDHHM0 (Year, DayOfYear, Hour, Minute)
    # 3. Download with retry logic (handle 404s for missing data)
    # 4. Save with original filename
    # 5. Log success/failures
```

**Task**: Download 2 weeks of 2024 data (~2,000 images) for testing
- Suggested period: September 2024 (active hurricane season)
- Output: `data/raw/noaa_test_2024/`

---

#### Step 1.2: Train YOLO11 on Current Dataset
**Script**: `training/train_yolo11.py`

```python
from ultralytics import YOLO

def train_hurricane_detector(data_yaml='data/raw/Hurricane.v3i.yolov8/data.yaml',
                             epochs=100):
    """
    Train YOLO11 on existing annotated dataset

    Args:
        data_yaml: Path to YOLO dataset config
        epochs: Training epochs (100-200 recommended)

    Returns:
        Trained model
    """
    # Model selection:
    # - yolo11n.pt: Nano (fast, 80-85% mAP, 2-3 hours training)
    # - yolo11s.pt: Small (balanced, 85-90% mAP, 3-4 hours)
    # - yolo11m.pt: Medium (accurate, 90-95% mAP, 4-6 hours)

    model = YOLO('yolo11n.pt')  # Start with Nano for speed

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=678,
        batch=16,  # Adjust based on GPU memory
        device='mps',  # 'cuda' for NVIDIA, 'cpu' for CPU-only
        patience=20,  # Early stopping
        save=True,
        augment=True,  # Critical for small datasets

        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        mosaic=1.0
    )

    # Evaluate on validation set
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    return model
```

**Expected Performance**:
- **With 822 images**: mAP50 ~80-90% (acceptable for auto-annotation)
- **Training time**: 2-4 hours on M-series Mac / NVIDIA GPU
- **Output**: `training/trained_models/yolo11_hurricane_v1.pt`

**Quality Threshold**:
- ✅ Proceed if mAP50 > 75%
- ⚠️ Review if mAP50 = 60-75%
- ❌ Stop if mAP50 < 60% (need more training data first)

---

#### Step 1.3: Auto-Annotate Test Dataset
**Script**: `preprocessing/7_auto_annotate.py`

```python
def auto_annotate_images(model_path, image_dir, output_dir,
                         confidence_threshold=0.5):
    """
    Auto-annotate images using trained YOLO model

    Args:
        model_path: Path to trained YOLO model (.pt file)
        image_dir: Directory containing images to annotate
        output_dir: Where to save YOLO label files
        confidence_threshold: Minimum detection confidence (0.3-0.7)

    Returns:
        annotation_stats: Dict with counts and confidence distribution
    """
    model = YOLO(model_path)

    annotations = []

    for image_path in image_dir.glob('*.jpg'):
        results = model.predict(
            image_path,
            conf=confidence_threshold,
            iou=0.5
        )

        labels = []
        for box in results[0].boxes:
            # YOLO format: class x_center y_center width height (normalized)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            labels.append({
                'class': class_id,
                'bbox': [x_center, y_center, width, height],
                'confidence': confidence
            })

        # Save YOLO label file
        label_path = output_dir / f"{image_path.stem}.txt"
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{label['class']} {' '.join(map(str, label['bbox']))}\n")

        annotations.append({
            'filename': image_path.name,
            'detections': len(labels),
            'confidences': [l['confidence'] for l in labels]
        })

    return annotations
```

**Task**: Auto-annotate 2,000 test images
- Use confidence threshold = 0.5 (adjust based on YOLO performance)
- Output: `data/raw/noaa_test_2024/labels/`

---

#### Step 1.4: Quality Assurance
**Script**: `utils/qa_annotations.py`

```python
def quality_check_annotations(image_dir, label_dir, sample_size=100):
    """
    Manual QA tool for reviewing auto-annotations

    Args:
        image_dir: Directory with images
        label_dir: Directory with YOLO labels
        sample_size: Number of samples to review

    Returns:
        qa_report: Accuracy, precision, recall estimates
    """
    # Implementation:
    # 1. Randomly sample images
    # 2. Display image with predicted bboxes
    # 3. Ask user: Correct? (Y/N/Partial)
    # 4. If incorrect, allow manual correction
    # 5. Calculate approval rate

    # UI: Simple matplotlib visualization or Gradio interface

    # Metrics to track:
    # - True Positive rate (correct detections)
    # - False Positive rate (ghost detections)
    # - False Negative rate (missed storms)
```

**Task**: Manually review 100 random samples
- **Acceptance criteria**: >90% correct detections
- If <90%, adjust confidence threshold or retrain YOLO

---

### Phase 2: Large-Scale Expansion (15 hours)

#### Step 2.1: Identify Target Dates
**Research 2024 Hurricane Events**

Major Atlantic hurricanes to target:
- Hurricane Beryl: June 28 - July 9, 2024
- Hurricane Ernesto: August 12-20, 2024
- Hurricane Francine: September 9-12, 2024
- Hurricane Helene: September 23-27, 2024
- Hurricane Milton: October 5-10, 2024

**Recommendation**: Download ±3 days around each event for context

**Estimated data**:
- 5 hurricanes × 10 days × 144 images/day = ~7,200 images
- Focus on continuous sequences for better time-series modeling

---

#### Step 2.2: Download Targeted Dataset
**Script**: Extend `preprocessing/6_download_noaa_data.py`

```python
HURRICANE_EVENTS_2024 = {
    'beryl': ('2024-06-25', '2024-07-12'),
    'ernesto': ('2024-08-09', '2024-08-23'),
    'francine': ('2024-09-06', '2024-09-15'),
    'helene': ('2024-09-20', '2024-09-30'),
    'milton': ('2024-10-02', '2024-10-13')
}

def download_hurricane_periods(events_dict, output_base_dir):
    """Download images for specific hurricane periods"""
    for name, (start, end) in events_dict.items():
        output_dir = output_base_dir / name
        download_noaa_images(
            datetime.strptime(start, '%Y-%m-%d'),
            datetime.strptime(end, '%Y-%m-%d'),
            output_dir
        )
```

**Storage estimate**:
- ~7,200 images × 70 KB/image ≈ 500 MB

---

#### Step 2.3: Auto-Annotate Full Dataset
**Task**: Run auto-annotation on all downloaded images
- Use trained YOLO model from Phase 1
- Batch processing for efficiency
- Save confidence scores for later filtering

---

#### Step 2.4: Quality Control (Stratified Sampling)
**Strategy**: Review samples from different confidence ranges

```python
def stratified_qa(annotations, samples_per_bin=20):
    """
    Review samples across confidence spectrum

    Bins:
    - High confidence: 0.8-1.0 (expect high accuracy)
    - Medium confidence: 0.5-0.8 (potential errors)
    - Low confidence: 0.3-0.5 (likely errors, but might catch edge cases)
    """
    # Sample 20 images from each bin
    # Manually verify
    # Calculate accuracy per bin
```

**Task**: Review ~500 samples (10% of uncertain detections)
- Correct egregious errors
- Flag patterns for YOLO retraining

---

#### Step 2.5: Retrain YOLO on Expanded Dataset
**Iterative Improvement**

```python
def iterative_training(initial_dataset, auto_annotated_dataset,
                      qa_corrections, iterations=2):
    """
    Iteratively improve YOLO through active learning

    Round 1:
    - Train on 822 manual + 500 QA-corrected auto-annotations
    - mAP improves to ~85-90%

    Round 2:
    - Re-annotate full dataset with improved model
    - Add high-confidence samples to training set
    - mAP improves to ~90-95%
    """
```

**Expected outcome**: mAP50-95 > 90% on validation set

---

### Phase 3: Rerun Preprocessing Pipeline

Once expanded dataset is ready:

1. **Combine datasets**: Merge 2023 + 2024 annotations
2. **Extract YOLO features**: Run on ~8,000 images
3. **Extract thermal features**: Run on ~8,000 images
4. **Build sequences**: Expect 50-100 continuous sequences
5. **Track storms**: Expect 500-1,000+ storm tracks

**Expected improvements**:
- Longest track: 1,000+ frames (1 week+)
- Mean track length: 50-100 frames
- Multiple full hurricane lifecycles

---

## 🎯 Success Criteria

### Phase 1 (Validation)
- ✅ YOLO mAP50 > 75% on current dataset
- ✅ >90% QA approval on 100 test samples
- ✅ Auto-annotation runs successfully on 2,000 images

### Phase 2 (Expansion)
- ✅ Download 5,000-10,000 images from 2024
- ✅ Auto-annotate with >85% estimated accuracy
- ✅ Retrained YOLO achieves mAP50-95 > 90%

### Phase 3 (Integration)
- ✅ Combined dataset > 8,000 annotated images
- ✅ >500 storm tracks (≥3 frames)
- ✅ Longest track > 500 frames

---

## ⚠️ Risk Mitigation

### Risk 1: YOLO Underperforms (mAP < 75%)
**Mitigation**:
- Use data augmentation aggressively
- Try larger model (yolo11m instead of yolo11n)
- Consider transfer learning from weather detection models
- Manual annotation of additional key frames

### Risk 2: Auto-Annotations Have Systematic Errors
**Mitigation**:
- Thorough QA with stratified sampling
- Correct patterns manually
- Retrain iteratively
- Use ensemble models (multiple YOLO variants)

### Risk 3: NOAA Data Unavailable or Incomplete
**Mitigation**:
- Check data availability before bulk download
- Have backup date ranges
- Accept some missing frames (our pipeline handles gaps)

### Risk 4: Training Time Too Long
**Mitigation**:
- Use smaller YOLO model (yolo11n)
- Reduce epochs with early stopping
- Use cloud GPU (Google Colab, AWS, etc.)
- Train overnight

---

## 📋 Checklist

### Before Starting
- [ ] Verify NOAA CDN accessibility
- [ ] Check GPU availability (local or cloud)
- [ ] Estimate storage requirements (~1-5 GB)
- [ ] Install YOLO dependencies: `pip install ultralytics`

### Phase 1 Tasks
- [ ] Write `preprocessing/6_download_noaa_data.py`
- [ ] Download 2-week test dataset (Sep 2024)
- [ ] Write `training/train_yolo11.py`
- [ ] Train YOLO11 on 822 images
- [ ] Evaluate YOLO performance (check mAP)
- [ ] Write `preprocessing/7_auto_annotate.py`
- [ ] Auto-annotate test dataset
- [ ] Write `utils/qa_annotations.py`
- [ ] Manually QA 100 samples
- [ ] Calculate approval rate
- [ ] **Decision point**: Proceed to Phase 2 if >90% approval

### Phase 2 Tasks (if Phase 1 succeeds)
- [ ] Research 2024 hurricane dates
- [ ] Download 5,000-10,000 images (targeted periods)
- [ ] Auto-annotate full dataset
- [ ] QA 500 samples (stratified)
- [ ] Correct errors
- [ ] Retrain YOLO on expanded dataset
- [ ] Re-auto-annotate with improved model
- [ ] Final QA check

### Phase 3 Tasks
- [ ] Merge 2023 + 2024 datasets
- [ ] Rerun `1_combine_datasets.py`
- [ ] Rerun `2_extract_yolo_features.py`
- [ ] Rerun `3_extract_thermal_features.py`
- [ ] Rerun `4_build_sequences.py`
- [ ] Rerun `5_track_storms.py`
- [ ] Validate results (check track statistics)

---

## 📊 Expected Outcomes

### Before Expansion (Current)
- Images: 822
- Tracks: 94
- Longest track: 289 frames (~48 hours)
- Mean track length: 25.64 frames

### After Expansion (Projected)
- Images: 5,000-10,000
- Tracks: 500-1,000+
- Longest track: 1,000+ frames (1+ week)
- Mean track length: 50-100 frames

### Impact on Forecasting Models
- **Tree models**: More diverse training examples → better generalization
- **LSTM/Transformers**: Longer sequences → better temporal modeling
- **Overall**: Higher accuracy, better storm lifecycle understanding

---

## 🚀 When to Execute

**Current Status**: Preprocessing pipeline complete with 822 images

**Recommended Timing**:
- **Option A**: Complete current pipeline first (feature engineering, PCA, initial training), then expand
- **Option B**: Expand dataset now before continuing pipeline ⭐ **RECOMMENDED**

**Rationale for Option B**:
- Time-series models benefit greatly from more data
- Better to invest upfront than retrain later
- Longer storm tracks = better LSTM performance
- Current dataset is limiting factor for deep learning

---

## 💾 Storage Requirements

### Phase 1
- Test images (2 weeks): ~2,000 images × 70 KB = 140 MB
- YOLO model: ~6-30 MB (depending on model size)
- Annotations: ~1 MB
- **Total**: ~200 MB

### Phase 2
- Full dataset (2-3 months): ~10,000 images × 70 KB = 700 MB
- YOLO models (multiple versions): ~50 MB
- Annotations: ~5 MB
- **Total**: ~800 MB

### Phase 3
- Combined preprocessed data: ~50-100 MB (parquet files)
- Storm tracks: ~5-10 MB

**Grand Total**: ~1-1.5 GB

---

## ⏰ Time Estimate

| Phase | Task | Time |
|-------|------|------|
| **Phase 1** | | |
| | Download test data | 30 min |
| | Train YOLO11 | 2-3 hours |
| | Auto-annotate | 15 min |
| | Manual QA | 1-2 hours |
| | **Subtotal** | **~5 hours** |
| **Phase 2** | | |
| | Research + plan | 1 hour |
| | Download data | 2-3 hours |
| | Auto-annotate | 30 min |
| | QA 500 samples | 4-5 hours |
| | Retrain YOLO | 4-6 hours |
| | **Subtotal** | **~15 hours** |
| **Phase 3** | | |
| | Rerun preprocessing | 1-2 hours |
| | Validate results | 30 min |
| | **Subtotal** | **~2 hours** |
| **TOTAL** | | **~22 hours** |

**Note**: Most time is YOLO training (can run overnight) and manual QA (can be spread over days)

---

## 📝 Notes

- **YOLO Version**: Use YOLO11 (latest) for best performance
- **Confidence Threshold**: Start at 0.5, adjust based on QA results
- **Hurricane Season**: June-November in Atlantic
- **Data Format**: YOLO annotations are text files, one per image
- **Validation**: Always maintain a held-out validation set (never auto-annotate validation data)

---

*Created: 2025-11-18*
*Status: TODO (Deferred until after current pipeline completion)*
