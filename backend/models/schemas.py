"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class BboxData(BaseModel):
    """Bounding box data"""
    x: float = Field(..., description="Center x coordinate (normalized 0-1)")
    y: float = Field(..., description="Center y coordinate (normalized 0-1)")
    w: float = Field(..., description="Width (normalized 0-1)")
    h: float = Field(..., description="Height (normalized 0-1)")


class StormData(BaseModel):
    """Storm information for a frame"""
    track_id: int = Field(..., description="Storm tracking ID")
    sequence_id: str = Field(..., description="Sequence ID")
    frame_idx: int = Field(..., description="Frame index within sequence")
    bbox: BboxData = Field(..., description="Bounding box coordinates")


class PredictionData(BaseModel):
    """Prediction for a specific horizon"""
    x: Optional[float] = Field(None, description="Predicted x coordinate")
    y: Optional[float] = Field(None, description="Predicted y coordinate")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Model confidence (0-1)")
    exists: bool = Field(True, description="Whether prediction exists")


class GroundTruthData(BaseModel):
    """Ground truth for a specific horizon"""
    x: Optional[float] = Field(None, description="Actual x coordinate")
    y: Optional[float] = Field(None, description="Actual y coordinate")
    exists: bool = Field(..., description="Whether ground truth exists")


class ErrorMetrics(BaseModel):
    """Error metrics for a prediction"""
    error_pct: float = Field(..., description="Position error as % of image size")
    error_pixels: float = Field(..., description="Position error in pixels")
    euclidean_distance: float = Field(..., description="Euclidean distance")


class StormPredictions(BaseModel):
    """All predictions for a single storm"""
    track_id: int
    t1: Optional[PredictionData] = Field(None, alias="t+1")
    t3: Optional[PredictionData] = Field(None, alias="t+3")
    t6: Optional[PredictionData] = Field(None, alias="t+6")
    t12: Optional[PredictionData] = Field(None, alias="t+12")

    class Config:
        populate_by_name = True


class StormGroundTruth(BaseModel):
    """All ground truth for a single storm"""
    track_id: int
    t1: Optional[GroundTruthData] = Field(None, alias="t+1")
    t3: Optional[GroundTruthData] = Field(None, alias="t+3")
    t6: Optional[GroundTruthData] = Field(None, alias="t+6")
    t12: Optional[GroundTruthData] = Field(None, alias="t+12")

    class Config:
        populate_by_name = True


class StormErrors(BaseModel):
    """All error metrics for a single storm"""
    track_id: int
    t1: Optional[ErrorMetrics] = Field(None, alias="t+1")
    t3: Optional[ErrorMetrics] = Field(None, alias="t+3")
    t6: Optional[ErrorMetrics] = Field(None, alias="t+6")
    t12: Optional[ErrorMetrics] = Field(None, alias="t+12")

    class Config:
        populate_by_name = True


class FrameResponse(BaseModel):
    """Complete frame data response"""
    frame_idx: int = Field(..., description="Frame index (0 to total_frames-1)")
    timestamp: str = Field(..., description="Frame timestamp (ISO format)")
    sequence_id: str = Field(..., description="Sequence identifier")
    filename: str = Field(..., description="Image filename")
    image_url: str = Field(..., description="URL to fetch image")
    storms: List[StormData] = Field(..., description="List of storms in this frame")
    predictions: Dict[int, StormPredictions] = Field(..., description="Predictions by track_id")
    ground_truth: Dict[int, StormGroundTruth] = Field(..., description="Ground truth by track_id")
    errors: Dict[int, StormErrors] = Field(..., description="Error metrics by track_id")


class MetadataResponse(BaseModel):
    """Dataset metadata"""
    total_frames: int = Field(..., description="Total number of frames")
    sequences: List[str] = Field(..., description="List of sequence IDs")
    date_range: Dict[str, str] = Field(..., description="Start and end dates")
    horizons: List[str] = Field(..., description="Available prediction horizons")
    confidence_scores: Dict[str, float] = Field(..., description="Model confidence by horizon")
