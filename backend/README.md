# Hurricane Forecast API Backend

FastAPI backend for the interactive hurricane forecast visualization.

## Features

✅ **Frame Endpoints**: Get satellite imagery and storm data
✅ **Metadata API**: Dataset information and model confidence scores
✅ **Error Metrics**: Real-time prediction error calculations
✅ **Image Streaming**: Efficient JPEG delivery with caching
✅ **CORS Enabled**: Ready for frontend integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Run Development Server

```bash
# From project root
python -m backend.main

# Server will start at http://localhost:8000
```

### 3. Access API Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## API Endpoints

### GET `/`
Root endpoint with API information

### GET `/health`
Health check endpoint

### GET `/api/frames/metadata`
Get dataset metadata:
- Total frames
- Sequence IDs
- Date range
- Available horizons
- Model confidence scores

**Response:**
```json
{
  "total_frames": 642,
  "sequences": ["seq_006", "seq_007", "seq_008"],
  "date_range": {
    "start": "2023-10-17 12:00:00",
    "end": "2023-10-21 14:20:00"
  },
  "horizons": ["t+1", "t+3", "t+6", "t+12"],
  "confidence_scores": {
    "t+1": 0.8623,
    "t+3": 0.8173,
    "t+6": 0.7611,
    "t+12": 0.5949
  }
}
```

### GET `/api/frames/{frame_idx}`
Get complete frame data including:
- Storm bounding boxes
- Predictions for all horizons
- Ground truth data
- Error metrics

**Parameters:**
- `frame_idx` (int): Frame index (0 to 641)

**Response:**
```json
{
  "frame_idx": 100,
  "timestamp": "2023-10-18 08:30:00",
  "sequence_id": "seq_006",
  "filename": "...",
  "image_url": "/api/frames/100/image",
  "storms": [
    {
      "track_id": 42,
      "sequence_id": "seq_006",
      "frame_idx": 50,
      "bbox": {
        "x": 0.45,
        "y": 0.30,
        "w": 0.05,
        "h": 0.04
      }
    }
  ],
  "predictions": {
    "42": {
      "t+1": {
        "x": 0.451,
        "y": 0.301,
        "confidence": 0.8623,
        "exists": true
      }
    }
  },
  "ground_truth": {
    "42": {
      "t+1": {
        "x": 0.452,
        "y": 0.302,
        "exists": true
      }
    }
  },
  "errors": {
    "42": {
      "t+1": {
        "error_pct": 0.85,
        "error_pixels": 5.76,
        "euclidean_distance": 0.0085
      }
    }
  }
}
```

### GET `/api/frames/{frame_idx}/image`
Stream satellite JPEG image for frame

**Parameters:**
- `frame_idx` (int): Frame index (0 to 641)

**Response**: JPEG image with caching headers

## Architecture

```
backend/
├── main.py                    # FastAPI app & startup
├── routers/
│   └── frames.py              # Frame endpoints
├── models/
│   └── schemas.py             # Pydantic models
├── services/
│   └── data_service.py        # Data access layer
└── requirements.txt
```

## Data Flow

```
[Frontend Request]
       ↓
[FastAPI Router]
       ↓
[DataService]
       ↓
[VisualizationDataLoader]  (reuses existing code)
       ↓
[Response]
```

## Development

### Hot Reload
The server automatically reloads on code changes:

```bash
python -m backend.main  # --reload is enabled by default
```

### Testing with curl

```bash
# Get metadata
curl http://localhost:8000/api/frames/metadata

# Get frame data
curl http://localhost:8000/api/frames/0

# Get image
curl http://localhost:8000/api/frames/0/image -o frame_0.jpg
```

### Testing with Python

```python
import requests

# Get metadata
response = requests.get('http://localhost:8000/api/frames/metadata')
print(response.json())

# Get frame
response = requests.get('http://localhost:8000/api/frames/100')
frame_data = response.json()
print(f"Storms: {len(frame_data['storms'])}")
```

## Deployment

### Production Server

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Next Steps

- [ ] Add caching layer (Redis) for frequently accessed frames
- [ ] Implement WebSocket for real-time updates
- [ ] Add batch endpoints for loading multiple frames
- [ ] Add compression (gzip) for JSON responses
- [ ] Add rate limiting
- [ ] Add authentication (if needed)

## Notes

- The backend reuses the existing `visualization/forecast_viz.py` data loader
- Error metrics are calculated on-demand for each request
- Images are served directly from disk with caching headers
- CORS is enabled for `localhost:3000` (Next.js default port)
