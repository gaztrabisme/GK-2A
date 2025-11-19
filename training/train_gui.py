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
from sweep_engine import SweepEngine
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
        self.sweep_results = None
        self.sweep_running = False
        self.tuned_params = None  # Store hyperparameter tuning results

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
                choices=["Random Forest", "XGBoost", "LightGBM"],
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

        with gr.Row():
            tune_btn = gr.Button("🔧 Tune Hyperparameters", variant="secondary", size="lg")
            train_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")

        with gr.Row():
            n_trials_slider = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="Number of Tuning Trials (More trials = better optimization, but slower)",
                info="Recommended: 50-100 trials. Current: 50"
            )

        tuning_log = gr.Textbox(label="Tuning Log", lines=10, interactive=False)
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

                # Parse horizon values (check longer strings first to avoid substring matches)
                horizon_values = []
                for h in horizons:
                    if "t+12" in h:
                        horizon_values.append(12)
                    elif "t+6" in h:
                        horizon_values.append(6)
                    elif "t+3" in h:
                        horizon_values.append(3)
                    elif "t+1" in h:
                        horizon_values.append(1)

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

                        # Check if we have tuned hyperparameters
                        if state.tuned_params:
                            log += "Using tuned hyperparameters from previous tuning session\n\n"

                        # Train all models for this horizon (includes ensemble)
                        def progress_callback(msg):
                            nonlocal log
                            log += "  " + msg + "\n"

                        horizon_results = engine.train_all_models(
                            X_train, y_train, X_test, y_test,
                            feature_names=feature_names,
                            model_names=model_names,
                            target_name=target_col,
                            progress_callback=progress_callback,
                            enable_ensemble=True,
                            model_params_dict=state.tuned_params
                        )

                        # Store results with horizon keys
                        for model_name, result in horizon_results.items():
                            result_key = f"{model_name}_t{horizon}"
                            all_results[result_key] = result

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

        def run_hyperparameter_tuning(selected_features, selected_models, target, horizons, n_trials):
            """Run hyperparameter tuning before training"""
            try:
                if state.full_data is None:
                    return "", "Please load data first (Tab 1)", "❌ No data loaded"

                from hyperparameter_tuner import HyperparameterTuner
                from splits.three_way_split import three_way_sequence_split
                from horizon_data_builder import HorizonDataBuilder

                # Extract feature names
                feature_names = [f.split('] ')[1] for f in selected_features]

                # Determine target column
                if "Position" in target:
                    target_col = "x_center"
                elif "Size" in target:
                    target_col = "bbox_area"
                else:
                    target_col = "mean_color"

                # Parse horizons (use first horizon for tuning)
                horizon = 1  # Default to t+1
                for h in horizons:
                    if "t+12" in h:
                        horizon = 12
                    elif "t+6" in h:
                        horizon = 6
                    elif "t+3" in h:
                        horizon = 3
                    elif "t+1" in h:
                        horizon = 1
                        break

                log = "=" * 80 + "\n"
                log += "HYPERPARAMETER TUNING\n"
                log += "=" * 80 + "\n\n"
                log += f"Features: {len(feature_names)}\n"
                log += f"Target: {target_col}\n"
                log += f"Horizon: t+{horizon}\n"
                log += f"Models: {', '.join(selected_models)}\n"
                log += f"Trials per model: {n_trials}\n\n"

                # Build horizon dataset
                X, y, idx_curr, idx_fut, df_curr, df_fut = HorizonDataBuilder.build_horizon_dataset(
                    state.full_data,
                    feature_cols=feature_names,
                    target_col=target_col,
                    horizon=horizon
                )

                # 3-way split: train/val/test
                train_df, val_df, test_df = three_way_sequence_split(df_curr, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

                # Get train/val sequences
                train_sequences = train_df['sequence_id'].unique().tolist()
                val_sequences = val_df['sequence_id'].unique().tolist()

                # Split horizon data by sequences
                from horizon_data_builder import HorizonDataBuilder
                X_train_temp, y_train_temp, X_val_temp, y_val_temp = HorizonDataBuilder.split_by_sequences(
                    df_curr, df_fut, X, y, train_sequences, val_sequences
                )

                X_train, y_train = X_train_temp, y_train_temp
                X_val, y_val = X_val_temp, y_val_temp

                log += f"\nData split:\n"
                log += f"  Train: {len(X_train)} samples\n"
                log += f"  Val:   {len(X_val)} samples\n"
                log += f"  Test:  {len(X) - len(X_train) - len(X_val)} samples\n\n"

                # Map model names
                model_map = {"Random Forest": "random_forest", "XGBoost": "xgboost", "LightGBM": "lightgbm"}
                model_names = [model_map.get(m, m.lower().replace(' ', '_')) for m in selected_models]

                # Tune each model
                tuned_params = {}
                for model_name in model_names:
                    log += "=" * 80 + "\n"
                    log += f"Tuning {model_name}...\n"
                    log += "=" * 80 + "\n\n"

                    tuner = HyperparameterTuner(model_name)

                    def progress_callback(trial_num, best_r2, params):
                        nonlocal log
                        if trial_num % 10 == 0 or trial_num == 1:
                            log += f"  Trial {trial_num}: Best R² = {best_r2:.4f}\n"

                    result = tuner.tune(X_train, y_train, X_val, y_val, n_trials=n_trials, progress_callback=progress_callback)

                    tuned_params[model_name] = result['best_params']

                    log += f"\n✅ Best params for {model_name}:\n"
                    for param, value in result['best_params'].items():
                        log += f"    {param}: {value}\n"
                    log += f"  Validation R²: {result['best_r2']:.4f}\n\n"

                # Store tuned params in state
                state.tuned_params = tuned_params

                log += "=" * 80 + "\n"
                log += "TUNING COMPLETE\n"
                log += "=" * 80 + "\n"
                log += "Use 'Start Training' to train with optimized hyperparameters.\n"

                return log, f"✅ Tuned {len(tuned_params)} models! Ready to train."

            except Exception as e:
                import traceback
                return f"Error during tuning:\n{str(e)}\n\n{traceback.format_exc()}", f"❌ Tuning failed: {str(e)}"

        tune_btn.click(
            fn=run_hyperparameter_tuning,
            inputs=[feature_selector, models_selector, target_selector, horizons_selector, n_trials_slider],
            outputs=[tuning_log, training_status]
        )

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


# Tab 5: Parameter Sweep
def create_sweep_tab():
    """Create Tab 5: AutoML Parameter Sweep"""
    with gr.Tab("5. Parameter Sweep"):
        gr.Markdown("## 🔍 Automated Hyperparameter Search")
        gr.Markdown("Randomized search over split strategies, features, targets, and horizons to find optimal configurations.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Sweep Configuration")

                n_iterations = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=50,
                    step=10,
                    label="Number of Iterations",
                    info="How many random configurations to try"
                )

                sweep_split_strategies = gr.CheckboxGroup(
                    choices=["sequence", "date", "random"],
                    value=["sequence"],
                    label="Split Strategies to Try",
                    info="Which train/test split methods to test"
                )

                sweep_train_ratio = gr.Slider(
                    minimum=0.6,
                    maximum=0.9,
                    value=0.8,
                    step=0.05,
                    label="Train Ratio Range (will randomize ±10%)",
                    info="Base train/test ratio"
                )

                sweep_feature_strategies = gr.CheckboxGroup(
                    choices=["pca_only", "raw_only", "motion_temporal_only", "random_subset", "all_features"],
                    value=["pca_only", "raw_only", "motion_temporal_only"],
                    label="Feature Strategies",
                    info="Which feature selection strategies to test"
                )

                sweep_targets = gr.CheckboxGroup(
                    choices=["x_center", "y_center", "bbox_area", "mean_color"],
                    value=["x_center", "y_center"],
                    label="Prediction Targets",
                    info="Which variables to predict"
                )

                sweep_horizons = gr.CheckboxGroup(
                    choices=[1, 3, 6, 12],
                    value=[1, 3, 6],
                    label="Time Horizons",
                    info="Which forecasting horizons to test"
                )

                sweep_models = gr.CheckboxGroup(
                    choices=["random_forest", "xgboost", "lightgbm"],
                    value=["random_forest", "xgboost"],
                    label="Models to Try",
                    info="Which model types to test"
                )

                with gr.Row():
                    start_sweep_btn = gr.Button("🚀 Start Sweep", variant="primary")
                    stop_sweep_btn = gr.Button("⏹️ Stop", variant="stop")
                    export_sweep_btn = gr.Button("💾 Export Results")

                export_status = gr.Textbox(
                    label="Export Status",
                    value="",
                    interactive=False,
                    lines=2
                )

            with gr.Column(scale=2):
                gr.Markdown("### Live Progress")

                sweep_status = gr.Textbox(
                    label="Status",
                    value="Ready to start sweep",
                    interactive=False,
                    lines=1
                )

                sweep_progress = gr.Textbox(
                    label="Progress Log",
                    interactive=False,
                    lines=20,
                    value=""
                )

                gr.Markdown("### Top 10 Configurations")
                leaderboard_table = gr.DataFrame(
                    label="Leaderboard (sorted by Test R²)",
                    interactive=False
                )

                best_config_display = gr.JSON(
                    label="Best Configuration Details",
                    value={}
                )

        def run_sweep(n_iter, split_strats, train_ratio, feat_strats, targets, horizons, models):
            """Execute parameter sweep"""
            try:
                if state.full_data is None:
                    return "❌ Please load data first (Tab 1)", "", None, {}

                state.sweep_running = True

                log = "="*80 + "\n"
                log += "PARAMETER SWEEP STARTED\n"
                log += "="*80 + "\n\n"

                # Build config space
                config_space = {
                    'split_strategy': split_strats if split_strats else ['sequence'],
                    'train_ratio': [max(0.6, train_ratio - 0.1), min(0.9, train_ratio + 0.1)],
                    'features': {
                        'strategies': feat_strats if feat_strats else ['all_features'],
                        'min_features': 5,
                        'max_features': sum(len(v) for v in state.feature_cols.values())
                    },
                    'target': targets if targets else ['x_center'],
                    'horizon': horizons if horizons else [1, 3],
                    'model': models if models else ['random_forest']
                }

                log += f"Config space:\n"
                log += f"  Split strategies: {config_space['split_strategy']}\n"
                log += f"  Train ratio range: {config_space['train_ratio']}\n"
                log += f"  Feature strategies: {config_space['features']['strategies']}\n"
                log += f"  Targets: {config_space['target']}\n"
                log += f"  Horizons: {config_space['horizon']}\n"
                log += f"  Models: {config_space['model']}\n\n"

                # Create sweep engine
                engine = SweepEngine(state.full_data, state.feature_cols)

                # Progress callback
                iteration_count = [0]

                def progress_callback(i, best_r2, result):
                    iteration_count[0] = i + 1
                    nonlocal log

                    test_r2 = result.get('test_r2', np.nan)
                    error = result.get('error', None)

                    if error:
                        log += f"[{i+1}/{n_iter}] ✗ Error: {error}\n"
                    else:
                        log += f"[{i+1}/{n_iter}] Test R²={test_r2:.4f} (Best: {best_r2:.4f}) | "
                        log += f"Model={result['model']}, Horizon=t+{result['horizon']}, "
                        log += f"Features={result['n_features']}, Target={result['target']}\n"

                # Run sweep
                results_df = engine.run_sweep(
                    config_space,
                    n_iterations=int(n_iter),
                    progress_callback=progress_callback
                )

                state.sweep_results = results_df
                state.sweep_running = False

                log += "\n" + "="*80 + "\n"
                log += "SWEEP COMPLETE\n"
                log += "="*80 + "\n"
                log += f"Total iterations: {len(results_df)}\n"
                log += f"Best Test R²: {engine.best_r2:.4f}\n\n"

                # Get leaderboard
                leaderboard = engine.get_leaderboard(top_k=10)

                # Format for display
                display_cols = ['iteration', 'model', 'horizon', 'target', 'n_features',
                               'test_r2', 'test_rmse', 'train_r2', 'split_strategy']
                leaderboard_display = leaderboard[display_cols].copy()
                leaderboard_display['test_r2'] = leaderboard_display['test_r2'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "NaN")
                leaderboard_display['test_rmse'] = leaderboard_display['test_rmse'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "NaN")
                leaderboard_display['train_r2'] = leaderboard_display['train_r2'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "NaN")

                # Best config
                best_config_json = {
                    'model': engine.best_config.get('model', 'N/A'),
                    'horizon': engine.best_config.get('horizon', 'N/A'),
                    'target': engine.best_config.get('target', 'N/A'),
                    'split_strategy': engine.best_config.get('split_strategy', 'N/A'),
                    'train_ratio': f"{engine.best_config.get('train_ratio', 0):.2f}",
                    'n_features': len(engine.best_config.get('features', [])),
                    'test_r2': f"{engine.best_r2:.4f}"
                }

                status = f"✅ Sweep complete: {len(results_df)} iterations | Best R² = {engine.best_r2:.4f}"

                return status, log, leaderboard_display, best_config_json

            except Exception as e:
                import traceback
                state.sweep_running = False
                error_msg = f"Error during sweep:\n{str(e)}\n\n{traceback.format_exc()}"
                return f"❌ Error: {str(e)}", error_msg, None, {}

        def export_sweep_results():
            """Export sweep results to CSV"""
            if state.sweep_results is None:
                return "❌ No sweep results to export"

            try:
                output_path = Path(__file__).parent / "sweep_results.csv"
                state.sweep_results.to_csv(output_path, index=False)
                return f"✅ Exported {len(state.sweep_results)} results to {output_path}"
            except Exception as e:
                return f"❌ Export failed: {str(e)}"

        start_sweep_btn.click(
            fn=run_sweep,
            inputs=[n_iterations, sweep_split_strategies, sweep_train_ratio,
                   sweep_feature_strategies, sweep_targets, sweep_horizons, sweep_models],
            outputs=[sweep_status, sweep_progress, leaderboard_table, best_config_display]
        )

        export_sweep_btn.click(
            fn=export_sweep_results,
            outputs=[export_status]
        )


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
        create_sweep_tab()

        gr.Markdown("---")
        gr.Markdown("*Powered by Gradio | Data: GOES-18 ABI Satellite Imagery*")

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
