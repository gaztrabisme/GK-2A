/**
 * TypeScript types for Hurricane Forecast API
 */

export interface BboxData {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface StormData {
  track_id: number;
  sequence_id: string;
  frame_idx: number;
  bbox: BboxData;
}

export interface PredictionData {
  x: number;
  y: number;
  confidence: number;
  exists: boolean;
}

export interface GroundTruthData {
  x: number;
  y: number;
  exists: boolean;
}

export interface ErrorMetrics {
  error_pct: number;
  error_pixels: number;
  euclidean_distance: number;
}

export type Horizon = "t+1" | "t+3" | "t+6" | "t+12";

export interface StormPredictions {
  track_id: number;
  "t+1"?: PredictionData;
  "t+3"?: PredictionData;
  "t+6"?: PredictionData;
  "t+12"?: PredictionData;
}

export interface StormGroundTruth {
  track_id: number;
  "t+1"?: GroundTruthData;
  "t+3"?: GroundTruthData;
  "t+6"?: GroundTruthData;
  "t+12"?: GroundTruthData;
}

export interface StormErrors {
  track_id: number;
  "t+1"?: ErrorMetrics;
  "t+3"?: ErrorMetrics;
  "t+6"?: ErrorMetrics;
  "t+12"?: ErrorMetrics;
}

export interface FrameResponse {
  frame_idx: number;
  timestamp: string;
  sequence_id: string;
  filename: string;
  image_url: string;
  storms: StormData[];
  predictions: Record<number, StormPredictions>;
  ground_truth: Record<number, StormGroundTruth>;
  errors: Record<number, StormErrors>;
}

export interface MetadataResponse {
  total_frames: number;
  sequences: string[];
  date_range: {
    start: string;
    end: string;
  };
  horizons: string[];
  confidence_scores: Record<Horizon, number>;
}

// UI State Types
export interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

export interface AppState {
  // Frame state
  currentFrame: number;
  totalFrames: number;
  frameData: FrameResponse | null;
  metadata: MetadataResponse | null;
  isLoading: boolean;

  // UI state
  visibleHorizons: Set<Horizon>;
  showCurrent: boolean;
  showActual: boolean;
  isPlaying: boolean;
  playbackSpeed: number;

  // Actions
  setFrame: (frame: number) => void;
  setFrameData: (data: FrameResponse) => void;
  setMetadata: (data: MetadataResponse) => void;
  toggleHorizon: (horizon: Horizon) => void;
  toggleCurrent: () => void;
  toggleActual: () => void;
  togglePlayback: () => void;
  setPlaybackSpeed: (speed: number) => void;
  setLoading: (loading: boolean) => void;
}

// Color constants
export const HORIZON_COLORS: Record<Horizon, [number, number, number]> = {
  "t+1": [128, 0, 255], // Purple
  "t+3": [0, 255, 128], // Bright green
  "t+6": [255, 128, 255], // Pink
  "t+12": [255, 255, 0], // Yellow
};

export const CURRENT_COLOR: [number, number, number] = [255, 0, 255]; // Magenta
export const ACTUAL_COLOR: [number, number, number] = [255, 255, 255]; // White
