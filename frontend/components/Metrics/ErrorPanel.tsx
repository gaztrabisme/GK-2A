"use client";

import { useState } from "react";
import { useStore } from "@/lib/store";
import { Horizon, HORIZON_COLORS } from "@/lib/types";
import { ChevronDown, ChevronUp, AlertCircle } from "lucide-react";

export default function ErrorPanel() {
  const { frameData } = useStore();
  const [expandedStorms, setExpandedStorms] = useState<Set<number>>(new Set());

  if (!frameData || frameData.storms.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No storms in current frame</p>
      </div>
    );
  }

  const toggleStorm = (trackId: number) => {
    const newSet = new Set(expandedStorms);
    if (newSet.has(trackId)) {
      newSet.delete(trackId);
    } else {
      newSet.add(trackId);
    }
    setExpandedStorms(newSet);
  };

  const getErrorColor = (errorPct: number): string => {
    if (errorPct < 1.0) return "text-green-400";
    if (errorPct < 2.0) return "text-yellow-400";
    return "text-red-400";
  };

  const horizons: Horizon[] = ["t+1", "t+3", "t+6", "t+12"];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">
          Error Metrics
        </h3>
        <span className="text-xs text-gray-500">
          {frameData.storms.length} storm{frameData.storms.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Error Legend */}
      <div className="text-xs text-gray-500 space-y-1 pb-3 border-b border-gray-700">
        <p>Position error as % of image size:</p>
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            <span>&lt; 1.0% (~7 km)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
            <span>1.0% - 2.0% (7-14 km)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-red-400 rounded-full"></div>
            <span>&gt; 2.0% (&gt;14 km)</span>
          </div>
        </div>
      </div>

      {/* Storm Cards */}
      <div className="space-y-2">
        {frameData.storms.map((storm) => {
          const errors = frameData.errors[storm.track_id];
          const isExpanded = expandedStorms.has(storm.track_id);

          // Calculate average error
          const errorValues = horizons
            .map((h) => errors?.[h]?.error_pct)
            .filter((e): e is number => e !== undefined);

          const avgError =
            errorValues.length > 0
              ? errorValues.reduce((a, b) => a + b, 0) / errorValues.length
              : null;

          return (
            <div
              key={storm.track_id}
              className="bg-gray-700 rounded-lg overflow-hidden"
            >
              {/* Storm Header */}
              <button
                onClick={() => toggleStorm(storm.track_id)}
                className="w-full px-3 py-2 flex items-center justify-between hover:bg-gray-650 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-magenta-500 rounded-full"></div>
                  <span className="text-sm font-medium">
                    Storm {storm.track_id}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {avgError !== null && (
                    <span
                      className={`text-xs font-mono ${getErrorColor(avgError)}`}
                    >
                      {avgError.toFixed(2)}%
                    </span>
                  )}
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  )}
                </div>
              </button>

              {/* Storm Details (Expandable) */}
              {isExpanded && errors && (
                <div className="px-3 py-2 space-y-2 bg-gray-750 border-t border-gray-600">
                  {horizons.map((horizon) => {
                    const error = errors[horizon];
                    const color = HORIZON_COLORS[horizon];

                    if (!error) {
                      return (
                        <div
                          key={horizon}
                          className="flex items-center justify-between text-xs"
                        >
                          <div className="flex items-center gap-2">
                            <div
                              className="w-2 h-2 rounded-full"
                              style={{
                                backgroundColor: `rgb(${color.join(",")})`,
                              }}
                            ></div>
                            <span className="text-gray-400">{horizon}</span>
                          </div>
                          <span className="text-gray-600">N/A</span>
                        </div>
                      );
                    }

                    return (
                      <div
                        key={horizon}
                        className="space-y-1"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-2 h-2 rounded-full"
                              style={{
                                backgroundColor: `rgb(${color.join(",")})`,
                              }}
                            ></div>
                            <span className="text-gray-300">{horizon}</span>
                          </div>
                          <span
                            className={`font-mono ${getErrorColor(error.error_pct)}`}
                          >
                            {error.error_pct.toFixed(2)}%
                          </span>
                        </div>

                        {/* Error Details */}
                        <div className="ml-4 text-xs text-gray-500 space-y-0.5">
                          <div className="flex justify-between">
                            <span>Pixels:</span>
                            <span>{error.error_pixels.toFixed(1)}px</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Distance:</span>
                            <span>~{(error.error_pct * 7).toFixed(1)}km</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Expand/Collapse All */}
      <button
        onClick={() => {
          if (expandedStorms.size === frameData.storms.length) {
            setExpandedStorms(new Set());
          } else {
            setExpandedStorms(
              new Set(frameData.storms.map((s) => s.track_id))
            );
          }
        }}
        className="w-full py-2 text-xs text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 rounded transition-colors"
      >
        {expandedStorms.size === frameData.storms.length
          ? "Collapse All"
          : "Expand All"}
      </button>
    </div>
  );
}
