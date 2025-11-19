"use client";

import { useStore } from "@/lib/store";
import { Horizon, HORIZON_COLORS } from "@/lib/types";
import { Eye, EyeOff } from "lucide-react";

export default function LayerControls() {
  const {
    visibleHorizons,
    toggleHorizon,
    showCurrent,
    toggleCurrent,
    showActual,
    toggleActual,
    metadata,
  } = useStore();

  const horizons: Horizon[] = ["t+1", "t+3", "t+6", "t+12"];

  const getHorizonLabel = (horizon: Horizon): string => {
    const labels = {
      "t+1": "10 minutes",
      "t+3": "30 minutes",
      "t+6": "1 hour",
      "t+12": "2 hours",
    };
    return labels[horizon];
  };

  const getConfidence = (horizon: Horizon): number => {
    if (!metadata) return 0;
    return metadata.confidence_scores[horizon] * 100;
  };

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-gray-300">Layers</h3>

      {/* Current & Actual Toggles */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 cursor-pointer group">
          <input
            type="checkbox"
            checked={showCurrent}
            onChange={toggleCurrent}
            className="w-4 h-4 rounded border-gray-600 text-blue-600 focus:ring-blue-500 focus:ring-offset-gray-800"
          />
          <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
            Current Positions
          </span>
          <div className="ml-auto flex items-center gap-1">
            {showCurrent ? (
              <Eye className="w-3 h-3 text-blue-400" />
            ) : (
              <EyeOff className="w-3 h-3 text-gray-600" />
            )}
          </div>
        </label>

        <label className="flex items-center gap-2 cursor-pointer group">
          <input
            type="checkbox"
            checked={showActual}
            onChange={toggleActual}
            className="w-4 h-4 rounded border-gray-600 text-blue-600 focus:ring-blue-500 focus:ring-offset-gray-800"
          />
          <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
            Ground Truth Paths
          </span>
          <div className="ml-auto flex items-center gap-1">
            {showActual ? (
              <Eye className="w-3 h-3 text-blue-400" />
            ) : (
              <EyeOff className="w-3 h-3 text-gray-600" />
            )}
          </div>
        </label>
      </div>

      {/* Horizon Predictions */}
      <div className="pt-3 border-t border-gray-700">
        <div className="text-xs font-semibold text-gray-400 mb-3">
          Predictions
        </div>

        <div className="space-y-3">
          {horizons.map((horizon) => {
            const isVisible = visibleHorizons.has(horizon);
            const confidence = getConfidence(horizon);
            const color = HORIZON_COLORS[horizon];

            return (
              <div key={horizon} className="space-y-1">
                <label className="flex items-center gap-2 cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={isVisible}
                    onChange={() => toggleHorizon(horizon)}
                    className="w-4 h-4 rounded border-gray-600 text-purple-600 focus:ring-purple-500 focus:ring-offset-gray-800"
                  />

                  {/* Color indicator */}
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{
                      backgroundColor: `rgb(${color.join(",")})`,
                    }}
                  ></div>

                  <div className="flex-1">
                    <div className="text-sm text-gray-300 group-hover:text-white transition-colors">
                      {horizon}
                    </div>
                    <div className="text-xs text-gray-500">
                      {getHorizonLabel(horizon)}
                    </div>
                  </div>

                  <div className="text-right">
                    {isVisible ? (
                      <Eye className="w-3 h-3 text-purple-400" />
                    ) : (
                      <EyeOff className="w-3 h-3 text-gray-600" />
                    )}
                  </div>
                </label>

                {/* Confidence Bar */}
                {metadata && (
                  <div className="ml-6 space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">Confidence</span>
                      <span className="text-gray-400">{confidence.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${confidence}%`,
                          backgroundColor: `rgb(${color.join(",")})`,
                        }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Toggle All */}
      <div className="pt-3 border-t border-gray-700">
        <button
          onClick={() => {
            if (visibleHorizons.size === horizons.length) {
              // Hide all
              horizons.forEach((h) => {
                if (visibleHorizons.has(h)) toggleHorizon(h);
              });
            } else {
              // Show all
              horizons.forEach((h) => {
                if (!visibleHorizons.has(h)) toggleHorizon(h);
              });
            }
          }}
          className="w-full py-2 text-xs text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 rounded transition-colors"
        >
          {visibleHorizons.size === horizons.length ? "Hide All" : "Show All"}
        </button>
      </div>
    </div>
  );
}
