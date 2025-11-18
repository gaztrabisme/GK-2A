"""
Training GUI - Gradio Interface

Multi-tab interface for feature selection, model configuration, training, and results.
"""

import gradio as gr
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data_preparation import TrainingDataPreparator
from training_engine import TrainingEngine
from splits.split_strategies import DataSplitter
from horizon_data_builder import HorizonDataBuilder
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.metrics import format_metrics_table

# Global state
class AppState:
    def __init__(self):
        self.data_prep = None
        self.full_data = None
        self.feature_cols = None
        self.train_df = None
        self.test_df = None
        self.results = None

    def initialize(self):
        """Load and prepare data"""
        if self.data_prep is None:
            self.data_prep = TrainingDataPreparator()
            self.full_data = self.data_prep.load_all_features()
            self.full_data = self.data_prep.apply_pca_transforms(self.full_data)
            self.feature_cols = self.data_prep.get_feature_columns()

state = AppState()


# Tab 1: Data & Features
def load_data_tab():
    """Initialize data and show overview"""
    try:
        state.initialize()

        overview = f"""
        ## Dataset Overview

        - **Total Samples**: {len(state.full_data):,}
        - **Sequences**: {state.full_data['sequence_id'].nunique()}
        - **Tracks**: {state.full_data['track_id'].nunique()}
        - **Date Range**: {state.full_data['timestamp'].min()} to {state.full_data['timestamp'].max()}

        ## Available Features

        - **Spatial (raw)**: {len(state.feature_cols['spatial_raw'])} features
        - **Thermal (raw)**: {len(state.feature_cols['thermal_raw'])} features
        - **Motion (raw)**: {len(state.feature_cols['motion_raw'])} features
        - **Temporal (raw)**: {len(state.feature_cols['temporal_raw'])} features
        - **Thermal PCA**: {len(state.feature_cols['thermal_pca'])} components
        - **Spatial PCA**: {len(state.feature_cols['spatial_pca'])} components

        **Total Features**: {sum(len(v) for v in state.feature_cols.values())}
        """

        # Sample data preview
        preview_df = state.full_data.head(5)[['timestamp', 'sequence_id', 'track_id', 'x_center', 'y_center', 'mean_color', 'speed']]

        # Get choices for the checkbox group
        choices = get_feature_choices()

        return overview, preview_df, "✅ Data loaded successfully!", gr.update(choices=choices, value=[])

    except Exception as e:
        return f"Error loading data: {str(e)}", None, "❌ Failed to load data", gr.update()


def get_feature_choices():
    """Get all feature choices for checkboxes"""
    if state.feature_cols is None:
        return []

    choices = []

    # Group features
    for group_name, features in state.feature_cols.items():
        group_label = group_name.replace('_', ' ').title()
        for feat in features:
            choices.append(f"[{group_label}] {feat}")

    return choices


def create_data_tab():
    """Create Tab 1: Data & Features"""
    with gr.Tab("1. Data & Features"):
        gr.Markdown("## Load Data and Select Features")

        load_btn = gr.Button("Load Dataset", variant="primary")
        status_text = gr.Textbox(label="Status", interactive=False)

        overview_md = gr.Markdown("")
        preview_table = gr.DataFrame(label="Data Preview")

        gr.Markdown("---")
        gr.Markdown("### Feature Selection")
        gr.Markdown("Select features to use for training. Avoid mixing raw features with their corresponding PCA components.")

        feature_selector = gr.CheckboxGroup(
            label="Available Features",
            choices=[],  # Will be populated after loading
            value=[]
        )

        select_all_btn = gr.Button("Select All")
        clear_all_btn = gr.Button("Clear All")
        smart_select_btn = gr.Button("Smart Select (PCA + Motion + Temporal)", variant="secondary")

        selected_summary = gr.Textbox(label="Selected Features Summary", interactive=False)

        # Event handlers
        load_btn.click(
            fn=load_data_tab,
            outputs=[overview_md, preview_table, status_text, feature_selector]
        )

        def smart_select():
            """Select PCA components + motion + temporal (recommended)"""
            if state.feature_cols is None:
                return []

            selected = []
            for feat in state.feature_cols['thermal_pca']:
                selected.append(f"[Thermal Pca] {feat}")
            for feat in state.feature_cols['spatial_pca']:
                selected.append(f"[Spatial Pca] {feat}")
            for feat in state.feature_cols['motion_raw']:
                selected.append(f"[Motion Raw] {feat}")
            for feat in state.feature_cols['temporal_raw']:
                selected.append(f"[Temporal Raw] {feat}")

            return selected

        smart_select_btn.click(fn=smart_select, outputs=[feature_selector])
        select_all_btn.click(fn=get_feature_choices, outputs=[feature_selector])
        clear_all_btn.click(fn=lambda: [], outputs=[feature_selector])

        def update_summary(selected):
            if not selected:
                return "No features selected"
            return f"Selected {len(selected)} features:\n" + "\n".join(f"  - {f}" for f in selected[:10]) + \
                   (f"\n  ... and {len(selected)-10} more" if len(selected) > 10 else "")

        feature_selector.change(fn=update_summary, inputs=[feature_selector], outputs=[selected_summary])

        return feature_selector


# Tab 2: Model Configuration
def create_config_tab(feature_selector):
    """Create Tab 2: Model Configuration"""
    with gr.Tab("2. Model Configuration"):
        gr.Markdown("## Configure Training Parameters")

        with gr.Row():
            models_selector = gr.CheckboxGroup(
                label="Models to Train",
                choices=["Random Forest"],  # Only RF for now
                value=["Random Forest"]
            )

            target_selector = gr.Radio(
                label="Prediction Target",
                choices=["Position (x, y)", "Size (area)", "Intensity (mean_color)"],
                value="Position (x, y)"
            )

        with gr.Row():
            horizons_selector = gr.CheckboxGroup(
                label="Time Horizons",
                choices=["t+1 (10 min)", "t+3 (30 min)", "t+6 (1 hour)", "t+12 (2 hours)"],
                value=["t+1 (10 min)"]
            )

            lookback_slider = gr.Slider(
                label="Lookback Window (frames)",
                minimum=3,
                maximum=12,
                value=6,
                step=1
            )

        gr.Markdown("### Train/Test Split")

        with gr.Row():
            split_strategy = gr.Dropdown(
                label="Split Strategy",
                choices=["Sequence-based Temporal", "Date-based", "Random"],
                value="Sequence-based Temporal"
            )

            train_ratio_slider = gr.Slider(
                label="Train Ratio",
                minimum=0.6,
                maximum=0.9,
                value=0.8,
                step=0.05
            )

        preview_split_btn = gr.Button("Preview Split")
        split_info_text = gr.Textbox(label="Split Information", interactive=False, lines=5)

        def preview_split(strategy, ratio):
            """Preview train/test split"""
            try:
                if state.full_data is None:
                    return "Please load data first (Tab 1)"

                # Perform split
                if strategy == "Sequence-based Temporal":
                    train_df, test_df = DataSplitter.sequence_based_temporal_split(state.full_data, ratio)
                elif strategy == "Date-based":
                    train_df, test_df = DataSplitter.date_based_split(state.full_data, train_ratio=ratio)
                else:  # Random
                    train_df, test_df = DataSplitter.random_split(state.full_data, ratio)

                # Get info
                info = DataSplitter.get_split_info(train_df, test_df)

                # Format output
                output = f"""
                Train samples: {info['train_samples']} ({info['train_ratio']:.1%})
                Test samples: {info['test_samples']} ({1-info['train_ratio']:.1%})

                Train sequences: {info.get('train_sequences', 'N/A')}
                Test sequences: {info.get('test_sequences', 'N/A')}

                Train date range: {info.get('train_date_range', ('N/A', 'N/A'))[0]} to {info.get('train_date_range', ('N/A', 'N/A'))[1]}
                Test date range: {info.get('test_date_range', ('N/A', 'N/A'))[0]} to {info.get('test_date_range', ('N/A', 'N/A'))[1]}
                """

                # Store split for training
                state.train_df = train_df
                state.test_df = test_df

                return output

            except Exception as e:
                return f"Error: {str(e)}"

        preview_split_btn.click(
            fn=preview_split,
            inputs=[split_strategy, train_ratio_slider],
            outputs=[split_info_text]
        )

        return models_selector, target_selector, horizons_selector, lookback_slider, split_strategy, train_ratio_slider


# Tab 3: Training
def create_training_tab(feature_selector, models_selector, target_selector, horizons_selector):
    """Create Tab 3: Training"""
    with gr.Tab("3. Training"):
        gr.Markdown("## Train Models")

        config_summary = gr.Textbox(label="Configuration Summary", lines=8, interactive=False)
        train_btn = gr.Button("Start Training", variant="primary", size="lg")
        training_log = gr.Textbox(label="Training Log", lines=15, interactive=False)
        training_status = gr.Textbox(label="Status", interactive=False)

        def generate_config_summary(features, models, target, horizons):
            """Generate human-readable config summary"""
            return f"""
            Features: {len(features)} selected
            Models: {', '.join(models)}
            Target: {target}
            Horizons: {', '.join(horizons)}
            """

        # Update summary when inputs change
        for inp in [feature_selector, models_selector, target_selector, horizons_selector]:
            inp.change(
                fn=generate_config_summary,
                inputs=[feature_selector, models_selector, target_selector, horizons_selector],
                outputs=[config_summary]
            )

        def run_training(selected_features, selected_models, target, horizons):
            """Execute training with multi-horizon support"""
            try:
                if state.full_data is None:
                    return "Please load data first (Tab 1)", "❌ No data loaded"

                log = "="*80 + "\n"
                log += "MULTI-HORIZON TRAINING\n"
                log += "="*80 + "\n\n"

                # Extract feature names from selections
                feature_names = [f.split('] ')[1] for f in selected_features]

                # Check features exist
                available = set(state.full_data.columns)
                missing = [f for f in feature_names if f not in available]
                if missing:
                    return f"Error: Missing features: {missing}", "❌ Feature error"

                # Determine target column
                if "Position" in target:
                    target_col = "x_center"  # Simplified: just x for now
                    # Check for target leakage
                    if any('spatial' in f.lower() or 'x_center' in f or 'y_center' in f for f in feature_names):
                        log += "⚠️ WARNING: Predicting position with spatial features (includes x_center, y_center)!\n"
                        log += "   This may cause target leakage. Consider using only motion + temporal features.\n\n"
                elif "Size" in target:
                    target_col = "bbox_area"
                    if 'bbox_area' in feature_names or any('spatial' in f.lower() for f in feature_names):
                        log += "⚠️ WARNING: Predicting size with spatial features (includes bbox_area)!\n"
                        log += "   This may cause target leakage.\n\n"
                else:  # Intensity
                    target_col = "mean_color"
                    if 'mean_color' in feature_names or any('thermal' in f.lower() for f in feature_names):
                        log += "⚠️ WARNING: Predicting intensity with thermal features (includes mean_color)!\n"
                        log += "   This may cause target leakage.\n\n"

                log += f"Configuration:\n"
                log += f"  Features: {len(feature_names)}\n"
                log += f"  Target: {target_col}\n"
                log += f"  Horizons: {horizons}\n"
                log += f"  Models: {selected_models}\n\n"

                # Parse horizon values
                horizon_values = []
                for h in horizons:
                    if "t+1" in h:
                        horizon_values.append(1)
                    elif "t+3" in h:
                        horizon_values.append(3)
                    elif "t+6" in h:
                        horizon_values.append(6)
                    elif "t+12" in h:
                        horizon_values.append(12)

                if not horizon_values:
                    return "Please select at least one time horizon (Tab 2)", "❌ No horizons selected"

                # Get train/test sequences (FIXED: ensure no overlap)
                if state.train_df is None:
                    # Default sequence split if not previewed
                    train_df, test_df = DataSplitter.sequence_based_temporal_split(state.full_data, 0.8)
                else:
                    train_df = state.train_df
                    test_df = state.test_df

                # Get unique sequences from each split
                train_sequences_set = set(train_df['sequence_id'].unique())
                test_sequences_set = set(test_df['sequence_id'].unique())

                # Find overlap (sequences appearing in both)
                overlap = train_sequences_set & test_sequences_set

                if overlap:
                    log += f"⚠️ CRITICAL WARNING: {len(overlap)} sequences appear in both train and test: {overlap}\n"
                    log += f"   This indicates the split strategy divided individual sequences!\n"
                    log += f"   For time-series data, use 'Sequence-based Temporal' split instead.\n"
                    log += f"   Proceeding anyway, but results may have data leakage...\n\n"

                train_sequences = list(train_sequences_set)
                test_sequences = list(test_sequences_set)

                log += f"Train sequences: {train_sequences}\n"
                log += f"Test sequences: {test_sequences}\n\n"

                # Map model names
                model_map = {"Random Forest": "random_forest"}
                model_names = [model_map.get(m, m.lower().replace(' ', '_')) for m in selected_models]

                # Train for each horizon
                all_results = {}
                engine = TrainingEngine()

                for horizon in horizon_values:
                    log += "="*80 + "\n"
                    log += f"HORIZON t+{horizon} ({horizon * 10} minutes)\n"
                    log += "="*80 + "\n\n"

                    try:
                        # Build horizon-specific dataset
                        X, y, idx_curr, idx_fut, df_curr, df_fut = HorizonDataBuilder.build_horizon_dataset(
                            state.full_data,
                            feature_cols=feature_names,
                            target_col=target_col,
                            horizon=horizon
                        )

                        # Split into train/test
                        X_train, y_train, X_test, y_test = HorizonDataBuilder.split_by_sequences(
                            df_curr, df_fut, X, y,
                            train_sequences, test_sequences
                        )

                        log += f"\nDataset built:\n"
                        log += f"  Train: {X_train.shape[0]} samples\n"
                        log += f"  Test: {X_test.shape[0]} samples\n\n"

                        # Train models for this horizon
                        for model_name in model_names:
                            log += f"Training {model_name} for t+{horizon}...\n"

                            def progress_callback(msg):
                                nonlocal log
                                log += "  " + msg + "\n"

                            result = engine.train_single_model(
                                model_name, X_train, y_train, X_test, y_test,
                                feature_names=feature_names,
                                model_params=None
                            )

                            # Store with horizon key
                            result_key = f"{model_name}_t{horizon}"
                            all_results[result_key] = result

                            metrics = result['metrics']
                            log += f"  ✓ Test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}\n\n"

                    except Exception as e:
                        log += f"  ✗ Error for t+{horizon}: {str(e)}\n\n"
                        continue

                # Store results
                state.results = all_results

                log += "\n" + "="*80 + "\n"
                log += "ALL TRAINING COMPLETE\n"
                log += "="*80 + "\n\n"

                # Format metrics table
                metrics_dict = {}
                for key, res in all_results.items():
                    if 'metrics' in res:
                        metrics_dict[key] = res['metrics']

                if metrics_dict:
                    log += format_metrics_table(metrics_dict)

                return log, f"✅ Trained {len(all_results)} models across {len(horizon_values)} horizons!"

            except Exception as e:
                import traceback
                return f"Error during training:\n{str(e)}\n\n{traceback.format_exc()}", f"❌ Error: {str(e)}"

        train_btn.click(
            fn=run_training,
            inputs=[feature_selector, models_selector, target_selector, horizons_selector],
            outputs=[training_log, training_status]
        )


# Tab 4: Results
def create_results_tab():
    """Create Tab 4: Results"""
    with gr.Tab("4. Results"):
        gr.Markdown("## Training Results")

        refresh_btn = gr.Button("Refresh Results")
        results_table = gr.DataFrame(label="Metrics Comparison")
        results_summary = gr.Textbox(label="Summary", interactive=False, lines=10)

        def load_results():
            """Load and display results"""
            if state.results is None:
                return None, "No results available. Please run training first."

            # Convert to dataframe
            rows = []
            for model_name, result in state.results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    rows.append({
                        'Model': model_name,
                        'Train RMSE': f"{metrics['train_rmse']:.4f}",
                        'Test RMSE': f"{metrics['test_rmse']:.4f}",
                        'Train MAE': f"{metrics['train_mae']:.4f}",
                        'Test MAE': f"{metrics['test_mae']:.4f}",
                        'Train R²': f"{metrics['train_r2']:.4f}",
                        'Test R²': f"{metrics['test_r2']:.4f}",
                        'Train Time (s)': f"{metrics['train_time']:.2f}"
                    })

            df = pd.DataFrame(rows) if rows else None

            # Summary
            summary = f"Trained {len(state.results)} model(s)\n\n"
            if df is not None:
                best_model = df.loc[df['Test R²'].astype(float).idxmax()]
                summary += f"Best model (by Test R²): {best_model['Model']}\n"
                summary += f"  Test RMSE: {best_model['Test RMSE']}\n"
                summary += f"  Test R²: {best_model['Test R²']}"

            return df, summary

        refresh_btn.click(fn=load_results, outputs=[results_table, results_summary])


# Main App
def create_app():
    """Create the main Gradio app"""
    with gr.Blocks(title="Hurricane Forecasting Training GUI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🌀 Hurricane Storm Forecasting - Training Interface")
        gr.Markdown("Multi-horizon storm prediction using tree-based models on satellite thermal imagery")

        # Create all tabs
        feature_selector = create_data_tab()
        models_selector, target_selector, horizons_selector, lookback_slider, split_strategy, train_ratio_slider = create_config_tab(feature_selector)
        create_training_tab(feature_selector, models_selector, target_selector, horizons_selector)
        create_results_tab()

        gr.Markdown("---")
        gr.Markdown("*Powered by Gradio | Data: GOES-18 ABI Satellite Imagery*")

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
