"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { getMetadata, getFrame } from "@/lib/api";
import SatelliteViewer from "@/components/Map/SatelliteViewer";
import TimelineControls from "@/components/Controls/TimelineControls";
import LayerControls from "@/components/Controls/LayerControls";
import ErrorPanel from "@/components/Metrics/ErrorPanel";

export default function Home() {
  const { currentFrame, setFrameData, setMetadata, setLoading, frameData } =
    useStore();

  // Load metadata on mount
  useEffect(() => {
    async function loadMetadata() {
      try {
        const data = await getMetadata();
        setMetadata(data);
      } catch (error) {
        console.error("Failed to load metadata:", error);
      }
    }
    loadMetadata();
  }, [setMetadata]);

  // Load frame data when currentFrame changes
  useEffect(() => {
    async function loadFrame() {
      setLoading(true);
      try {
        const data = await getFrame(currentFrame);
        setFrameData(data);
      } catch (error) {
        console.error("Failed to load frame:", error);
        setLoading(false);
      }
    }
    loadFrame();
  }, [currentFrame, setFrameData, setLoading]);

  return (
    <main className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
            Hurricane Forecast
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            Interactive Storm Tracker
          </p>
        </div>

        {/* Timeline Controls */}
        <div className="p-4 border-b border-gray-700">
          <TimelineControls />
        </div>

        {/* Layer Controls */}
        <div className="p-4 border-b border-gray-700">
          <LayerControls />
        </div>

        {/* Error Metrics */}
        <div className="flex-1 overflow-y-auto p-4">
          {frameData && <ErrorPanel />}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 text-xs text-gray-500">
          <p>Data: GOES-18 ABI Full Disk</p>
          <p>Model: LightGBM Stacking Ensemble</p>
        </div>
      </aside>

      {/* Main Viewer */}
      <div className="flex-1 relative">
        <SatelliteViewer />
      </div>
    </main>
  );
}
