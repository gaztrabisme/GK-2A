/**
 * API client for Hurricane Forecast Backend
 */

import axios from "axios";
import { FrameResponse, MetadataResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Get dataset metadata
 */
export async function getMetadata(): Promise<MetadataResponse> {
  const response = await api.get<MetadataResponse>("/api/frames/metadata");
  return response.data;
}

/**
 * Get frame data for a specific frame index
 */
export async function getFrame(frameIdx: number): Promise<FrameResponse> {
  const response = await api.get<FrameResponse>(`/api/frames/${frameIdx}`);
  return response.data;
}

/**
 * Get image URL for a frame
 */
export function getImageUrl(frameIdx: number): string {
  return `${API_BASE_URL}/api/frames/${frameIdx}/image`;
}

/**
 * Preload image for caching
 */
export function preloadImage(frameIdx: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve();
    img.onerror = () => reject(new Error(`Failed to load image ${frameIdx}`));
    img.src = getImageUrl(frameIdx);
  });
}

/**
 * Batch preload multiple frames
 */
export async function preloadFrames(frameIndices: number[]): Promise<void> {
  await Promise.all(frameIndices.map((idx) => preloadImage(idx)));
}

export default {
  getMetadata,
  getFrame,
  getImageUrl,
  preloadImage,
  preloadFrames,
};
