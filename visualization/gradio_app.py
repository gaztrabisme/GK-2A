"""
Interactive Gradio visualization for hurricane forecast predictions.
Shows satellite imagery with YOLO bboxes, predicted trajectories, and error metrics.
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.forecast_viz import VisualizationDataLoader

# Color scheme (BGR format for OpenCV) - chosen to contrast with thermal imagery
# Thermal imagery uses: blue (cold) -> green -> yellow -> red (hot)
# So we use: magenta, white, purple, bright green, pink
COLORS_BGR = {
    "current": (255, 0, 255),  # Magenta - current position
    "actual": (255, 255, 255),  # White - actual future path
    "t+1": (255, 0, 128),  # Purple - t+1 prediction
    "t+3": (128, 255, 0),  # Bright green - t+3 prediction
    "t+6": (255, 128, 255),  # Pink - t+6 prediction
    "t+12": (0, 255, 255),  # Cyan - t+12 prediction
}

# RGB colors for Plotly (convert from BGR)
COLORS_RGB = {
    "current": "rgb(255, 0, 255)",  # Magenta
    "actual": "rgb(255, 255, 255)",  # White
    "t+1": "rgb(128, 0, 255)",  # Purple
    "t+3": "rgb(0, 255, 128)",  # Bright green
    "t+6": "rgb(255, 128, 255)",  # Pink
    "t+12": "rgb(255, 255, 0)",  # Cyan
}

HORIZONS = ["t+1", "t+3", "t+6", "t+12"]


class ForecastVisualizer:
    """Main visualization class"""

    def __init__(self):
        """Initialize data loader"""
        print("Loading visualization data...")
        self.loader = VisualizationDataLoader()
        self.total_frames = self.loader.get_total_frames()
        print(f"✓ Loaded {self.total_frames} frames")

    def denormalize_coords(self, x, y=None, img_width=678, img_height=678):
        """Convert normalized [0,1] coords to pixel coordinates"""
        if y is None:
            # Single value (width or height)
            return int(x * img_width)
        return int(x * img_width), int(y * img_height)

    def draw_bbox(self, img, bbox, color, label="", thickness=2):
        """Draw bounding box on image"""
        h, w = img.shape[:2]

        x_center, y_center = self.denormalize_coords(bbox["x"], bbox["y"], w, h)
        width = self.denormalize_coords(bbox["w"], img_width=w)
        height = self.denormalize_coords(bbox["h"], img_height=h)

        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if label:
            cv2.putText(
                img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    def draw_prediction_dot(self, img, x, y, color, confidence, label="", size=5):
        """Draw prediction point with confidence ring"""
        h, w = img.shape[:2]
        px, py = self.denormalize_coords(x, y, w, h)

        # Uncertainty ring - larger when less confident
        uncertainty_radius = int(30 * (1 - confidence))
        if uncertainty_radius > 2:
            cv2.circle(img, (px, py), uncertainty_radius, color, 1)

        # Prediction dot
        cv2.circle(img, (px, py), size, color, -1)

        if label:
            cv2.putText(
                img, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    def draw_trajectory_line(
        self, img, start_x, start_y, end_x, end_y, color, dashed=False
    ):
        """Draw trajectory line (solid or dashed)"""
        h, w = img.shape[:2]
        p1 = self.denormalize_coords(start_x, start_y, w, h)
        p2 = self.denormalize_coords(end_x, end_y, w, h)

        if dashed:
            # Draw dashed line
            dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            dash_length = 8
            gap_length = 4

            if dist > 0:
                num_dashes = int(dist / (dash_length + gap_length))
                for i in range(num_dashes):
                    start_frac = i * (dash_length + gap_length) / dist
                    end_frac = (i * (dash_length + gap_length) + dash_length) / dist

                    x1 = int(p1[0] + start_frac * (p2[0] - p1[0]))
                    y1 = int(p1[1] + start_frac * (p2[1] - p1[1]))
                    x2 = int(p1[0] + end_frac * (p2[0] - p1[0]))
                    y2 = int(p1[1] + end_frac * (p2[1] - p1[1]))

                    cv2.line(img, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.line(img, p1, p2, color, 2)

    def calculate_error_metrics(self, predicted, actual):
        """Calculate prediction error metrics"""
        if actual is None or not actual.get("exists", False):
            return None

        dx = predicted["x"] - actual["x"]
        dy = predicted["y"] - actual["y"]
        position_error = np.sqrt(dx**2 + dy**2)
        position_error_pct = position_error * 100  # % of image

        # Direction error (if we have velocity)
        direction_error = None
        if "direction" in predicted and "direction" in actual:
            direction_error = abs(predicted["direction"] - actual["direction"])
            if direction_error > 180:
                direction_error = 360 - direction_error

        return {
            "position_error_pct": position_error_pct,
            "direction_error": direction_error,
        }

    def render_frame(
        self,
        frame_idx,
        show_current=True,
        show_actual=True,
        show_t1=True,
        show_t3=True,
        show_t6=True,
        show_t12=True,
    ):
        """Render a single frame with all visualizations"""
        # Get frame data
        frame_data = self.loader.get_frame(frame_idx)

        if frame_data is None:
            # Create blank image with error message
            img = np.zeros((678, 678, 3), dtype=np.uint8)
            cv2.putText(
                img,
                "Frame not found",
                (200, 339),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return img, "Frame not found", {}

        # Get image (already loaded by the data loader)
        img = frame_data["image"].copy()

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prepare error metrics
        all_errors = {}

        # For each storm in the frame
        for storm in frame_data["storms"]:
            track_id = storm["track_id"]

            # Draw current position bbox
            if show_current:
                label = f"Storm {track_id}"
                self.draw_bbox(
                    img, storm["bbox"], COLORS_BGR["current"], label, thickness=2
                )

            # Draw predictions and actual paths
            current_x = storm["bbox"]["x"]
            current_y = storm["bbox"]["y"]

            storm_errors = {}

            # Get predictions and ground truth for this storm
            storm_predictions = frame_data["predictions"].get(track_id, {})
            storm_ground_truth = frame_data["ground_truth_future"].get(track_id, {})

            # Horizons to show
            horizons_to_show = []
            if show_t1:
                horizons_to_show.append("t+1")
            if show_t3:
                horizons_to_show.append("t+3")
            if show_t6:
                horizons_to_show.append("t+6")
            if show_t12:
                horizons_to_show.append("t+12")

            last_pred_x, last_pred_y = current_x, current_y
            last_actual_x, last_actual_y = current_x, current_y

            for horizon in horizons_to_show:
                if horizon not in storm_predictions:
                    continue

                pred = storm_predictions[horizon]
                actual = storm_ground_truth.get(horizon)
                confidence = pred["confidence"]

                # Draw predicted trajectory line (dashed)
                self.draw_trajectory_line(
                    img,
                    last_pred_x,
                    last_pred_y,
                    pred["x"],
                    pred["y"],
                    COLORS_BGR[horizon],
                    dashed=True,
                )

                # Draw prediction point with confidence ring
                label = f"{horizon} ({confidence:.0%})"
                self.draw_prediction_dot(
                    img, pred["x"], pred["y"], COLORS_BGR[horizon], confidence, label
                )

                last_pred_x, last_pred_y = pred["x"], pred["y"]

                # Draw actual path (solid)
                if actual and actual.get("exists", False) and show_actual:
                    self.draw_trajectory_line(
                        img,
                        last_actual_x,
                        last_actual_y,
                        actual["x"],
                        actual["y"],
                        COLORS_BGR["actual"],
                        dashed=False,
                    )

                    # Draw actual position dot (smaller)
                    self.draw_prediction_dot(
                        img,
                        actual["x"],
                        actual["y"],
                        COLORS_BGR["actual"],
                        1.0,
                        "",
                        size=3,
                    )

                    last_actual_x, last_actual_y = actual["x"], actual["y"]

                # Calculate error metrics
                error = self.calculate_error_metrics(pred, actual)
                if error:
                    storm_errors[horizon] = error

            if storm_errors:
                all_errors[track_id] = storm_errors

        # Create frame info text
        info_text = self._create_info_text(frame_data, all_errors)

        return img, info_text, all_errors

    def _create_info_text(self, frame_data, all_errors):
        """Create formatted info text for display"""
        lines = []
        lines.append(f"**Frame {frame_data['frame_idx']} / {self.total_frames - 1}**")
        lines.append(f"**Time:** {frame_data['timestamp']}")
        lines.append(f"**Sequence:** {frame_data['sequence_id']}")
        lines.append(f"**Storms in frame:** {len(frame_data['storms'])}")
        lines.append("")

        # Error metrics
        if all_errors:
            lines.append("**Prediction Errors:**")
            for track_id, errors in all_errors.items():
                lines.append(f"\n*Storm {track_id}:*")
                for horizon, metrics in errors.items():
                    pos_err = metrics["position_error_pct"]
                    lines.append(f"  - {horizon}: {pos_err:.2f}% position error")
        else:
            lines.append("*No ground truth available for error calculation*")

        return "\n".join(lines)

    def render_frame_plotly(
        self,
        frame_idx,
        show_current=True,
        show_actual=True,
        show_t1=True,
        show_t3=True,
        show_t6=True,
        show_t12=True,
    ):
        """Render frame using Plotly for interactive zoom"""
        # Get frame data
        frame_data = self.loader.get_frame(frame_idx)

        if frame_data is None:
            # Return empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="Frame not found",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color="red"),
            )
            return fig

        # Get image and convert to RGB
        img = frame_data["image"].copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Create plotly figure with image as trace (not layout)
        fig = go.Figure()

        # Add satellite image as an Image trace
        fig.add_trace(go.Image(z=img_rgb))

        # For each storm, add overlays
        for storm in frame_data["storms"]:
            track_id = storm["track_id"]

            # Get predictions and ground truth
            storm_predictions = frame_data["predictions"].get(track_id, {})
            storm_ground_truth = frame_data["ground_truth_future"].get(track_id, {})

            # Current position bbox
            if show_current:
                bbox = storm["bbox"]
                x_center = bbox["x"] * w
                y_center = bbox["y"] * h
                box_w = bbox["w"] * w
                box_h = bbox["h"] * h

                x0 = x_center - box_w / 2
                y0 = y_center - box_h / 2
                x1 = x_center + box_w / 2
                y1 = y_center + box_h / 2

                # Draw rectangle
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color=COLORS_RGB["current"], width=2),
                    name=f"Storm {track_id}",
                )

                # Add label
                fig.add_annotation(
                    x=x0,
                    y=y0 - 5,
                    text=f"Storm {track_id}",
                    showarrow=False,
                    font=dict(color=COLORS_RGB["current"], size=10),
                    bgcolor="rgba(0,0,0,0.5)",
                )

            # Prepare trajectory data
            current_x = storm["bbox"]["x"] * w
            current_y = storm["bbox"]["y"] * h

            # Horizons to show
            horizons_to_show = []
            if show_t1:
                horizons_to_show.append("t+1")
            if show_t3:
                horizons_to_show.append("t+3")
            if show_t6:
                horizons_to_show.append("t+6")
            if show_t12:
                horizons_to_show.append("t+12")

            # Predicted path
            pred_x = [current_x]
            pred_y = [current_y]
            pred_labels = []
            pred_colors = []

            # Actual path
            actual_x = [current_x]
            actual_y = [current_y]

            for horizon in horizons_to_show:
                if horizon in storm_predictions:
                    pred = storm_predictions[horizon]
                    px = pred["x"] * w
                    py = pred["y"] * h
                    pred_x.append(px)
                    pred_y.append(py)
                    pred_labels.append(f"{horizon} ({pred['confidence']:.0%})")
                    pred_colors.append(COLORS_RGB[horizon])

                    # Actual path
                    if horizon in storm_ground_truth:
                        actual = storm_ground_truth[horizon]
                        if actual.get("exists", False):
                            ax = actual["x"] * w
                            ay = actual["y"] * h
                            actual_x.append(ax)
                            actual_y.append(ay)

            # Draw predicted trajectory (dashed line)
            if len(pred_x) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=pred_x,
                        y=pred_y,
                        mode="lines+markers",
                        line=dict(dash="dash", width=2, color=COLORS_RGB["t+6"]),
                        marker=dict(
                            size=8,
                            color=pred_colors if pred_colors else COLORS_RGB["t+6"],
                        ),
                        name=f"Predicted path (Storm {track_id})",
                        text=pred_labels,
                        hovertemplate="%{text}<br>x=%{x:.0f}, y=%{y:.0f}",
                        showlegend=False,
                    )
                )

            # Draw actual path (solid line)
            if len(actual_x) > 1 and show_actual:
                fig.add_trace(
                    go.Scatter(
                        x=actual_x,
                        y=actual_y,
                        mode="lines+markers",
                        line=dict(width=2, color=COLORS_RGB["actual"]),
                        marker=dict(size=4, color=COLORS_RGB["actual"]),
                        name=f"Actual path (Storm {track_id})",
                        hovertemplate="Actual<br>x=%{x:.0f}, y=%{y:.0f}",
                        showlegend=False,
                    )
                )

        # Update layout for image display
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

        fig.update_layout(
            width=900,
            height=900,
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="closest",
            title=f"Frame {frame_idx}: {frame_data['timestamp']}",
            xaxis=dict(scaleanchor="y", scaleratio=1),
        )

        return fig


def create_gradio_interface():
    """Create and launch Gradio interface"""

    # Initialize visualizer
    viz = ForecastVisualizer()

    # Create interface
    with gr.Blocks(
        title="Hurricane Forecast Visualization", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 🌀 Hurricane Forecast Visualization")
        gr.Markdown(
            "Interactive visualization of hurricane predictions from stacking ensemble models"
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Main plot display with zoom capability
                plot_output = gr.Plot(
                    label="Satellite Image with Predictions (Zoom Enabled)"
                )

                # Timeline slider
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=viz.total_frames - 1,
                    step=1,
                    value=0,
                    label="Frame Timeline",
                    info="Drag to navigate through time",
                )

                # Playback controls
                with gr.Row():
                    prev_btn = gr.Button("⏮️ Previous", size="sm")
                    play_btn = gr.Button("▶️ Play", size="sm")
                    next_btn = gr.Button("⏭️ Next", size="sm")

                gr.Markdown("💡 **Use mouse to zoom and pan the image above**")

            with gr.Column(scale=1):
                # Frame info and metrics
                gr.Markdown("### Frame Information")
                info_output = gr.Markdown()

                gr.Markdown("### Display Controls")

                # Toggle controls
                show_current = gr.Checkbox(
                    label="Current Position (Magenta)", value=True
                )
                show_actual = gr.Checkbox(label="Actual Path (White)", value=True)

                gr.Markdown("**Predictions:**")
                show_t1 = gr.Checkbox(label="t+1 (10 min) - Purple", value=True)
                show_t3 = gr.Checkbox(label="t+3 (30 min) - Bright Green", value=True)
                show_t6 = gr.Checkbox(label="t+6 (1 hour) - Pink", value=True)
                show_t12 = gr.Checkbox(label="t+12 (2 hours) - Cyan", value=True)

                gr.Markdown("### Legend")
                gr.Markdown("""
                - **Magenta box**: Current storm position (YOLO bbox)
                - **White line**: Actual future path
                - **Colored dots**: Predicted positions
                - **Dashed lines**: Predicted trajectories
                - **Rings around dots**: Uncertainty (larger = less confident)

                **Position Error %:**
                - Error is measured as % of image width
                - 1% error ≈ 6.78 pixels ≈ 75-100 km (on full Earth disk)
                - Example: "1.12% position error" = ~7.6 pixels = ~80-100 km off
                """)

        # Function to update frame
        def update_frame(frame_idx, show_cur, show_act, t1, t3, t6, t12):
            fig = viz.render_frame_plotly(
                int(frame_idx),
                show_current=show_cur,
                show_actual=show_act,
                show_t1=t1,
                show_t3=t3,
                show_t6=t6,
                show_t12=t12,
            )

            # Get frame data and calculate errors
            frame_data = viz.loader.get_frame(int(frame_idx))
            if frame_data:
                # Calculate error metrics for each storm
                all_errors = {}

                for storm in frame_data["storms"]:
                    track_id = storm["track_id"]
                    storm_predictions = frame_data["predictions"].get(track_id, {})
                    storm_ground_truth = frame_data["ground_truth_future"].get(
                        track_id, {}
                    )

                    storm_errors = {}
                    for horizon in ["t+1", "t+3", "t+6", "t+12"]:
                        if (
                            horizon in storm_predictions
                            and horizon in storm_ground_truth
                        ):
                            pred = storm_predictions[horizon]
                            actual = storm_ground_truth[horizon]
                            error = viz.calculate_error_metrics(pred, actual)
                            if error:
                                storm_errors[horizon] = error

                    if storm_errors:
                        all_errors[track_id] = storm_errors

                info = viz._create_info_text(frame_data, all_errors)
            else:
                info = "Frame not found"

            return fig, info

        # Navigation functions
        def go_prev(current):
            new_idx = max(0, current - 1)
            return new_idx

        def go_next(current):
            new_idx = min(viz.total_frames - 1, current + 1)
            return new_idx

        # Connect controls
        inputs = [
            frame_slider,
            show_current,
            show_actual,
            show_t1,
            show_t3,
            show_t6,
            show_t12,
        ]
        outputs = [plot_output, info_output]

        frame_slider.change(update_frame, inputs=inputs, outputs=outputs)
        show_current.change(update_frame, inputs=inputs, outputs=outputs)
        show_actual.change(update_frame, inputs=inputs, outputs=outputs)
        show_t1.change(update_frame, inputs=inputs, outputs=outputs)
        show_t3.change(update_frame, inputs=inputs, outputs=outputs)
        show_t6.change(update_frame, inputs=inputs, outputs=outputs)
        show_t12.change(update_frame, inputs=inputs, outputs=outputs)

        prev_btn.click(go_prev, inputs=frame_slider, outputs=frame_slider)
        next_btn.click(go_next, inputs=frame_slider, outputs=frame_slider)

        # Load initial frame
        demo.load(update_frame, inputs=inputs, outputs=outputs)

        gr.Markdown("""
        ---
        **Prediction Confidence Scores:**
        - Purple (t+1): 86.2% confidence - LightGBM Stacking
        - Bright Green (t+3): 81.7% confidence - LightGBM Stacking
        - Pink (t+6): 76.1% confidence - LightGBM Stacking
        - Cyan (t+12): 59.5% confidence - LightGBM

        *Confidence scores from ensemble model test R² values*
        """)

    return demo


if __name__ == "__main__":
    print("Starting Hurricane Forecast Visualization...")
    demo = create_gradio_interface()
    demo.launch(share=True)
