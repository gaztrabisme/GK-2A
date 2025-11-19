"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";

export default function TimelineControls() {
  const {
    currentFrame,
    totalFrames,
    setFrame,
    isPlaying,
    togglePlayback,
    playbackSpeed,
    setPlaybackSpeed,
    frameData,
  } = useStore();

  // Playback loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setFrame((currentFrame + 1) % totalFrames);
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, currentFrame, totalFrames, playbackSpeed, setFrame]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setFrame(Math.min(currentFrame + 1, totalFrames - 1));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setFrame(Math.max(currentFrame - 1, 0));
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlayback();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [currentFrame, totalFrames, setFrame, togglePlayback]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFrame(parseInt(e.target.value));
  };

  const skipBackward = () => {
    setFrame(Math.max(currentFrame - 10, 0));
  };

  const skipForward = () => {
    setFrame(Math.min(currentFrame + 10, totalFrames - 1));
  };

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-gray-300">Timeline</h3>

      {/* Playback Controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={skipBackward}
          className="p-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          title="Skip backward (10 frames)"
        >
          <SkipBack className="w-4 h-4" />
        </button>

        <button
          onClick={togglePlayback}
          className="p-2 bg-blue-600 hover:bg-blue-500 rounded transition-colors flex-shrink-0"
          title={isPlaying ? "Pause (Space)" : "Play (Space)"}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4" />
          )}
        </button>

        <button
          onClick={skipForward}
          className="p-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          title="Skip forward (10 frames)"
        >
          <SkipForward className="w-4 h-4" />
        </button>

        <div className="text-xs text-gray-400 ml-auto">
          {currentFrame + 1} / {totalFrames}
        </div>
      </div>

      {/* Timeline Slider */}
      <div className="space-y-2">
        <input
          type="range"
          min="0"
          max={totalFrames - 1}
          value={currentFrame}
          onChange={handleSliderChange}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />

        {/* Date Labels */}
        {frameData && (
          <div className="flex justify-between text-xs text-gray-500">
            <span>Oct 17</span>
            <span>Oct 19</span>
            <span>Oct 21</span>
          </div>
        )}
      </div>

      {/* Playback Speed */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400">Playback Speed</label>
          <span className="text-xs text-gray-300">{playbackSpeed} FPS</span>
        </div>

        <input
          type="range"
          min="1"
          max="30"
          value={playbackSpeed}
          onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
        />

        <div className="flex justify-between text-xs text-gray-600">
          <span>1x</span>
          <span>15x</span>
          <span>30x</span>
        </div>
      </div>

      {/* Keyboard Shortcuts */}
      <div className="pt-2 border-t border-gray-700">
        <div className="text-xs text-gray-500 space-y-1">
          <div className="flex justify-between">
            <span>← / →</span>
            <span className="text-gray-400">Navigate frames</span>
          </div>
          <div className="flex justify-between">
            <span>Space</span>
            <span className="text-gray-400">Play / Pause</span>
          </div>
        </div>
      </div>
    </div>
  );
}
