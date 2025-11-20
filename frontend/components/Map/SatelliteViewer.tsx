"use client";

import { useMemo } from "react";
import DeckGL from "@deck.gl/react";
import { BitmapLayer, ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import { OrthographicView } from "@deck.gl/core";
import { useStore } from "@/lib/store";
import { getImageUrl } from "@/lib/api";
import {
  HORIZON_COLORS,
  CURRENT_COLOR,
  ACTUAL_COLOR,
  Horizon,
} from "@/lib/types";

const INITIAL_VIEW_STATE = {
  target: [339, 339, 0],
  zoom: 0,
};

export default function SatelliteViewer() {
  const {
    currentFrame,
    frameData,
    visibleHorizons,
    showCurrent,
    showActual,
    isLoading,
  } = useStore();

  // Generate layers
  const layers = useMemo(() => {
    if (!frameData) return [];

    const layerList = [];

    // 1. Satellite Image Layer (base)
    layerList.push(
      new BitmapLayer({
        id: "satellite-image",
        image: getImageUrl(currentFrame),
        bounds: [0, 678, 678, 0], // [left, bottom, right, top] in pixel coordinates
        desaturate: 0,
        transparentColor: [0, 0, 0, 0],
        tintColor: [255, 255, 255],
      })
    );

    // 2. Current Storm Positions (Magenta boxes)
    if (showCurrent && frameData.storms.length > 0) {
      const currentPositions = frameData.storms.map((storm) => ({
        position: [
          storm.bbox.x * 678, // Convert normalized to pixels
          storm.bbox.y * 678, // YOLO format: (0,0) is top-left
        ],
        radius: Math.max(storm.bbox.w, storm.bbox.h) * 678 * 0.5, // Use bbox size
        color: CURRENT_COLOR,
        trackId: storm.track_id,
      }));

      layerList.push(
        new ScatterplotLayer({
          id: "current-storms",
          data: currentPositions,
          pickable: true,
          opacity: 0.8,
          stroked: true,
          filled: false,
          radiusScale: 1,
          radiusMinPixels: 5,
          radiusMaxPixels: 100,
          lineWidthMinPixels: 2,
          getPosition: (d) => d.position,
          getRadius: (d) => d.radius,
          getFillColor: [0, 0, 0, 0],
          getLineColor: (d) => d.color,
          getLineWidth: 2,
        })
      );
    }

    // 3. Ground Truth Paths (White dashed lines with endpoint circle)
    if (showActual) {
      const allHorizons: Horizon[] = ["t+1", "t+3", "t+6", "t+12"];

      frameData.storms.forEach((storm) => {
        const trackId = storm.track_id;
        const groundTruth = frameData.ground_truth[trackId];

        if (!groundTruth) return;

        // Build path from current position through all horizons
        const pathPoints: [number, number][] = [
          [storm.bbox.x * 678, storm.bbox.y * 678], // Current position
        ];

        allHorizons.forEach((horizon) => {
          const gt = groundTruth[horizon];
          if (gt && gt.exists) {
            pathPoints.push([gt.x * 678, gt.y * 678]);
          }
        });

        if (pathPoints.length > 1) {
          // Draw white dashed line path (same style as predictions)
          layerList.push(
            new PathLayer({
              id: `actual-path-${trackId}`,
              data: [{ path: pathPoints }],
              pickable: false,
              widthScale: 1,
              widthMinPixels: 1,
              getDashArray: [5, 3], // Dashed line like predictions
              getPath: (d) => d.path,
              getColor: ACTUAL_COLOR,
              getWidth: 1.5,
            })
          );

          // Draw white circle at final endpoint only
          const endpoint = pathPoints[pathPoints.length - 1];
          layerList.push(
            new ScatterplotLayer({
              id: `actual-endpoint-${trackId}`,
              data: [
                {
                  position: endpoint,
                  color: ACTUAL_COLOR,
                },
              ],
              pickable: true,
              opacity: 0.9,
              stroked: true,
              filled: true,
              radiusScale: 1,
              radiusMinPixels: 6,
              radiusMaxPixels: 15,
              lineWidthMinPixels: 2,
              getPosition: (d) => d.position,
              getRadius: 8,
              getFillColor: (d) => [...d.color, 200],
              getLineColor: (d) => d.color,
              getLineWidth: 2,
            })
          );
        }
      });
    }

    // 4. Predicted Paths (Colored by horizon)
    frameData.storms.forEach((storm) => {
      const trackId = storm.track_id;
      const predictions = frameData.predictions[trackId];

      if (!predictions) return;

      Array.from(visibleHorizons).forEach((horizon) => {
        const pred = predictions[horizon];

        if (!pred || !pred.exists) return;

        // Draw prediction point
        layerList.push(
          new ScatterplotLayer({
            id: `pred-${horizon}-${trackId}`,
            data: [
              {
                position: [pred.x * 678, pred.y * 678],
                color: HORIZON_COLORS[horizon],
                confidence: pred.confidence,
              },
            ],
            pickable: true,
            opacity: 0.9,
            stroked: true,
            filled: true,
            radiusScale: 1,
            radiusMinPixels: 6,
            radiusMaxPixels: 15,
            lineWidthMinPixels: 2,
            getPosition: (d) => d.position,
            getRadius: (d) => 8 + (1 - d.confidence) * 20, // Larger radius = less confident
            getFillColor: (d) => [...d.color, 200],
            getLineColor: (d) => d.color,
            getLineWidth: 2,
          })
        );

        // Draw path from current to prediction
        const pathPoints: [number, number][] = [
          [storm.bbox.x * 678, storm.bbox.y * 678],
          [pred.x * 678, pred.y * 678],
        ];

        layerList.push(
          new PathLayer({
            id: `pred-path-${horizon}-${trackId}`,
            data: [{ path: pathPoints }],
            pickable: false,
            widthScale: 1,
            widthMinPixels: 1,
            getDashArray: [5, 3], // Dashed line
            getPath: (d) => d.path,
            getColor: HORIZON_COLORS[horizon],
            getWidth: 1.5,
          })
        );
      });
    });

    return layerList;
  }, [
    frameData,
    currentFrame,
    visibleHorizons,
    showCurrent,
    showActual,
  ]);

  return (
    <div className="relative w-full h-full">
      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-300">Loading frame {currentFrame}...</p>
          </div>
        </div>
      )}

      {/* Deck.gl Map */}
      <DeckGL
        views={new OrthographicView({ id: "ortho" })}
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
        getTooltip={({ object }) => {
          if (object && object.trackId) {
            return {
              html: `<div class="bg-gray-800 text-white p-2 rounded shadow-lg">
                <strong>Storm ${object.trackId}</strong>
              </div>`,
              style: {
                backgroundColor: "transparent",
                fontSize: "0.8em",
              },
            };
          }
          return null;
        }}
      />

      {/* Frame Info Overlay */}
      {frameData && (
        <div className="absolute top-4 left-4 bg-gray-800 bg-opacity-90 text-white p-3 rounded-lg shadow-lg">
          <div className="text-xs space-y-1">
            <div className="font-semibold">
              Frame {frameData.frame_idx + 1} / 642
            </div>
            <div className="text-gray-300">{frameData.timestamp}</div>
            <div className="text-gray-400">
              Sequence: {frameData.sequence_id}
            </div>
            <div className="text-gray-400">
              Storms: {frameData.storms.length}
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-800 bg-opacity-90 text-white p-3 rounded-lg shadow-lg">
        <div className="text-xs font-semibold mb-2">Legend</div>
        <div className="space-y-1 text-xs">
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 border-2 rounded"
              style={{ borderColor: `rgb(${CURRENT_COLOR.join(",")})` }}
            ></div>
            <span>Current Position</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-1"
              style={{ backgroundColor: `rgb(${ACTUAL_COLOR.join(",")})` }}
            ></div>
            <span>Actual Path</span>
          </div>
          {(["t+1", "t+3", "t+6", "t+12"] as Horizon[]).map((horizon) => (
            <div key={horizon} className="flex items-center gap-2">
              <div
                className="w-4 h-4 rounded-full"
                style={{
                  backgroundColor: `rgb(${HORIZON_COLORS[horizon].join(",")})`,
                }}
              ></div>
              <span>
                {horizon} ({horizon === "t+1" ? "10min" : horizon === "t+3" ? "30min" : horizon === "t+6" ? "1hr" : "2hr"})
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
