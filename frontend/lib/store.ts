/**
 * Global state management with Zustand
 */

import { create } from "zustand";
import { AppState, Horizon, FrameResponse, MetadataResponse } from "./types";

export const useStore = create<AppState>((set) => ({
  // Initial state
  currentFrame: 0,
  totalFrames: 642,
  frameData: null,
  metadata: null,
  isLoading: false,

  visibleHorizons: new Set<Horizon>(["t+1", "t+3", "t+6", "t+12"]),
  showCurrent: true,
  showActual: true,
  isPlaying: false,
  playbackSpeed: 10, // FPS

  // Actions
  setFrame: (frame: number) =>
    set((state) => ({
      currentFrame: Math.max(0, Math.min(frame, state.totalFrames - 1)),
    })),

  setFrameData: (data: FrameResponse) =>
    set({ frameData: data, isLoading: false }),

  setMetadata: (data: MetadataResponse) =>
    set({ metadata: data, totalFrames: data.total_frames }),

  toggleHorizon: (horizon: Horizon) =>
    set((state) => {
      const newSet = new Set(state.visibleHorizons);
      if (newSet.has(horizon)) {
        newSet.delete(horizon);
      } else {
        newSet.add(horizon);
      }
      return { visibleHorizons: newSet };
    }),

  toggleCurrent: () => set((state) => ({ showCurrent: !state.showCurrent })),

  toggleActual: () => set((state) => ({ showActual: !state.showActual })),

  togglePlayback: () => set((state) => ({ isPlaying: !state.isPlaying })),

  setPlaybackSpeed: (speed: number) => set({ playbackSpeed: speed }),

  setLoading: (loading: boolean) => set({ isLoading: loading }),
}));
