"""
Frame data endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from backend.models.schemas import FrameResponse, MetadataResponse
from backend.services.data_service import get_data_service


router = APIRouter(prefix="/api/frames", tags=["frames"])


@router.get("/metadata", response_model=MetadataResponse)
async def get_metadata():
    """
    Get dataset metadata including total frames, date range, and confidence scores.
    """
    service = get_data_service()
    return service.get_metadata()


@router.get("/{frame_idx}", response_model=FrameResponse)
async def get_frame(frame_idx: int):
    """
    Get complete frame data including storms, predictions, ground truth, and errors.

    Args:
        frame_idx: Frame index (0 to total_frames-1)

    Returns:
        Complete frame data
    """
    service = get_data_service()

    try:
        frame_data = service.get_frame_data(frame_idx)

        # Add image URL
        frame_data["image_url"] = f"/api/frames/{frame_idx}/image"

        return frame_data

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading frame: {str(e)}")


@router.get("/{frame_idx}/image")
async def get_frame_image(frame_idx: int):
    """
    Get satellite image for frame.

    Args:
        frame_idx: Frame index

    Returns:
        JPEG image file
    """
    service = get_data_service()

    try:
        image_path = service.get_image_path(frame_idx)

        if not image_path or not Path(image_path).exists():
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(
            image_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=86400"  # Cache for 24 hours
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")
