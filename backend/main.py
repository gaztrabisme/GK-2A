"""
FastAPI Backend for Hurricane Forecast Visualization

Main application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.routers import frames


# Create FastAPI app
app = FastAPI(
    title="Hurricane Forecast API",
    description="Backend API for interactive hurricane forecast visualization",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(frames.router)


# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "message": "Hurricane Forecast API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


# Health check
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    # Run with: python backend/main.py
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
