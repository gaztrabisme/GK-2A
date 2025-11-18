"""
PCA Analysis with Second Derivative Elbow Detection

Implements grouped PCA on thermal and spatial features using the second derivative
method to automatically select the optimal number of principal components.
"""

import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class PCAAnalyzer:
    def __init__(
        self,
        metadata_path="features/metadata.yml",
        output_dir="pca"
    ):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = yaml.safe_load(f)

        self.pca_groups = self.metadata['pca_groups']

    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load spatial and thermal features"""
        print("Loading feature data...")

        spatial_df = pd.read_parquet('data/processed/features/spatial_features.parquet')
        thermal_df = pd.read_parquet('data/processed/features/thermal_features.parquet')

        print(f"  Spatial: {len(spatial_df)} samples")
        print(f"  Thermal: {len(thermal_df)} samples")

        return spatial_df, thermal_df

    def find_elbow_second_derivative(
        self,
        explained_variance_ratio: np.ndarray,
        min_components: int = 2
    ) -> int:
        """
        Find elbow point using second derivative method.

        Algorithm:
        1. Calculate first derivative (rate of change)
        2. Calculate second derivative (acceleration)
        3. Find index of maximum absolute second derivative
        4. Return number of components before the sharpest drop

        Args:
            explained_variance_ratio: Array of explained variance per component
            min_components: Minimum number of components to keep

        Returns:
            Optimal number of components
        """
        n_components = len(explained_variance_ratio)

        if n_components <= min_components:
            return n_components

        # Convert to percentage for easier interpretation
        var_pct = explained_variance_ratio * 100

        # First derivative (rate of change)
        first_deriv = np.diff(var_pct)

        # Second derivative (acceleration)
        second_deriv = np.diff(first_deriv)

        # Find the point with maximum absolute second derivative
        # This is where the curve changes most sharply
        elbow_idx = np.argmax(np.abs(second_deriv))

        # Return number of components (elbow_idx + 2 because we lost 2 points in diff operations)
        # But we want components BEFORE the elbow (where most variance is)
        n_selected = elbow_idx + 2

        # Ensure we keep at least min_components
        n_selected = max(n_selected, min_components)

        # Don't exceed available components
        n_selected = min(n_selected, n_components)

        return n_selected

    def visualize_elbow(
        self,
        explained_variance_ratio: np.ndarray,
        n_selected: int,
        group_name: str
    ):
        """
        Create visualization of elbow detection.

        Args:
            explained_variance_ratio: Variance ratios per component
            n_selected: Selected number of components
            group_name: Name of feature group (for title)
        """
        n_components = len(explained_variance_ratio)
        var_pct = explained_variance_ratio * 100

        # Calculate cumulative variance
        cumvar_pct = np.cumsum(var_pct)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'PCA Analysis: {group_name}', fontsize=16, fontweight='bold')

        # 1. Scree plot (individual variance)
        ax1 = axes[0, 0]
        ax1.plot(range(1, n_components + 1), var_pct, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(n_selected, color='red', linestyle='--', linewidth=2,
                    label=f'Selected: {n_selected} PCs')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance (%)', fontsize=12)
        ax1.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # 2. Cumulative variance
        ax2 = axes[0, 1]
        ax2.plot(range(1, n_components + 1), cumvar_pct, 'go-', linewidth=2, markersize=8)
        ax2.axvline(n_selected, color='red', linestyle='--', linewidth=2,
                    label=f'At PC{n_selected}: {cumvar_pct[n_selected-1]:.2f}%')
        ax2.axhline(95, color='orange', linestyle=':', linewidth=1.5, label='95% threshold')
        ax2.set_xlabel('Principal Component', fontsize=12)
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # 3. First derivative
        ax3 = axes[1, 0]
        first_deriv = np.diff(var_pct)
        ax3.plot(range(2, n_components + 1), first_deriv, 'mo-', linewidth=2, markersize=6)
        ax3.axvline(n_selected, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Principal Component', fontsize=12)
        ax3.set_ylabel('Rate of Change (% per PC)', fontsize=12)
        ax3.set_title('First Derivative (Rate of Change)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)

        # 4. Second derivative
        ax4 = axes[1, 1]
        second_deriv = np.diff(first_deriv)
        elbow_idx = np.argmax(np.abs(second_deriv))
        ax4.plot(range(3, n_components + 1), second_deriv, 'co-', linewidth=2, markersize=6)
        ax4.plot(elbow_idx + 3, second_deriv[elbow_idx], 'r*', markersize=20,
                 label=f'Elbow at PC{elbow_idx + 2}')
        ax4.axvline(n_selected, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Principal Component', fontsize=12)
        ax4.set_ylabel('Acceleration', fontsize=12)
        ax4.set_title('Second Derivative (Elbow Detection)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.legend(fontsize=10)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / 'analysis' / f'pca_elbow_{group_name}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved visualization: {plot_path}")
        plt.close()

    def analyze_pca_group(
        self,
        df: pd.DataFrame,
        group_name: str,
        group_config: Dict
    ) -> Dict:
        """
        Perform PCA analysis on a feature group.

        Args:
            df: DataFrame containing features
            group_name: Name of PCA group
            group_config: Configuration from metadata.yml

        Returns:
            Dictionary with PCA results
        """
        print(f"\n{'='*80}")
        print(f"PCA ANALYSIS: {group_name.upper()}")
        print(f"{'='*80}")

        # Extract features
        feature_cols = group_config['input_features']
        print(f"\nInput features ({len(feature_cols)}): {', '.join(feature_cols)}")

        # Check which columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]

        if missing_cols:
            print(f"  Warning: Missing columns: {missing_cols}")

        if not available_cols:
            raise ValueError(f"No features found for {group_name}")

        X = df[available_cols].copy()

        # Handle missing values
        n_missing = X.isnull().sum().sum()
        if n_missing > 0:
            print(f"  Warning: {n_missing} missing values, filling with column means")
            X = X.fillna(X.mean())

        print(f"  Samples: {len(X)}")

        # Standardize features
        print("\nStandardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit PCA with all components
        n_features = len(available_cols)
        pca = PCA(n_components=n_features)
        pca.fit(X_scaled)

        # Print explained variance
        print("\nExplained variance per component:")
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            cumvar = np.sum(pca.explained_variance_ratio_[:i+1])
            print(f"  PC{i+1}: {var_ratio*100:6.2f}% (cumulative: {cumvar*100:6.2f}%)")

        # Find elbow
        n_selected = self.find_elbow_second_derivative(pca.explained_variance_ratio_)
        cumvar_at_elbow = np.sum(pca.explained_variance_ratio_[:n_selected]) * 100

        print(f"\n{'='*80}")
        print(f"ELBOW DETECTION RESULT")
        print(f"{'='*80}")
        print(f"Selected components: {n_selected}")
        print(f"Variance explained: {cumvar_at_elbow:.2f}%")

        # Create visualization
        self.visualize_elbow(pca.explained_variance_ratio_, n_selected, group_name)

        # Refit PCA with selected components
        pca_final = PCA(n_components=n_selected)
        pca_final.fit(X_scaled)

        # Get loadings (feature contributions to each PC)
        loadings = pd.DataFrame(
            pca_final.components_.T,
            columns=[f'PC{i+1}_{group_config["output_prefix"].split("_")[1]}'
                    for i in range(n_selected)],
            index=available_cols
        )

        print(f"\nTop feature loadings for each PC:")
        for col in loadings.columns:
            abs_loadings = loadings[col].abs().sort_values(ascending=False)
            top_3 = abs_loadings.head(3)
            print(f"\n  {col}:")
            for feat, val in top_3.items():
                print(f"    {feat}: {val:.3f} ({loadings.loc[feat, col]:.3f})")

        return {
            'scaler': scaler,
            'pca': pca_final,
            'n_components': n_selected,
            'explained_variance_ratio': pca_final.explained_variance_ratio_,
            'cumulative_variance': cumvar_at_elbow,
            'loadings': loadings,
            'feature_names': available_cols,
            'output_prefix': group_config['output_prefix']
        }

    def run_pca_analysis(self):
        """Run PCA analysis on all configured groups"""
        print("="*80)
        print("GROUPED PCA ANALYSIS WITH ELBOW DETECTION")
        print("="*80)

        # Load features
        spatial_df, thermal_df = self.load_features()

        results = {}

        # Analyze thermal PCA
        if 'thermal_pca' in self.pca_groups:
            results['thermal_pca'] = self.analyze_pca_group(
                thermal_df,
                'thermal_pca',
                self.pca_groups['thermal_pca']
            )

        # Analyze spatial PCA
        if 'spatial_pca' in self.pca_groups:
            results['spatial_pca'] = self.analyze_pca_group(
                spatial_df,
                'spatial_pca',
                self.pca_groups['spatial_pca']
            )

        return results

    def save_results(self, results: Dict):
        """Save PCA transformers, scalers, and configuration"""
        print("\n" + "="*80)
        print("SAVING PCA RESULTS")
        print("="*80)

        # Save transformers and scalers
        transformers_dir = self.output_dir / 'transformers'
        transformers_dir.mkdir(parents=True, exist_ok=True)

        for group_name, group_results in results.items():
            # Save scaler
            scaler_path = transformers_dir / f'scaler_{group_name.split("_")[0]}.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(group_results['scaler'], f)
            print(f"  Saved: {scaler_path}")

            # Save PCA transformer
            pca_path = transformers_dir / f'pca_{group_name.split("_")[0]}.pkl'
            with open(pca_path, 'wb') as f:
                pickle.dump(group_results['pca'], f)
            print(f"  Saved: {pca_path}")

        # Save loadings
        analysis_dir = self.output_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)

        for group_name, group_results in results.items():
            loadings_path = analysis_dir / f'pca_loadings_{group_name.split("_")[0]}.csv'
            group_results['loadings'].to_csv(loadings_path)
            print(f"  Saved: {loadings_path}")

        # Generate and save PCA config
        config_dir = self.output_dir / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)

        pca_config = {
            'pca_results': {}
        }

        for group_name, group_results in results.items():
            pca_config['pca_results'][group_name] = {
                'n_components': int(group_results['n_components']),
                'cumulative_variance': float(group_results['cumulative_variance']),
                'explained_variance_ratio': [float(x) for x in group_results['explained_variance_ratio']],
                'feature_names': group_results['feature_names'],
                'output_prefix': group_results['output_prefix'],
                'component_names': list(group_results['loadings'].columns)
            }

        config_path = config_dir / 'pca_config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(pca_config, f, indent=2, default_flow_style=False)
        print(f"  Saved: {config_path}")

        print("\n✅ All PCA results saved successfully!")

    def print_summary(self, results: Dict):
        """Print final summary"""
        print("\n" + "="*80)
        print("PCA ANALYSIS SUMMARY")
        print("="*80)

        for group_name, group_results in results.items():
            print(f"\n{group_name.upper()}:")
            print(f"  Input features: {len(group_results['feature_names'])}")
            print(f"  Selected PCs: {group_results['n_components']}")
            print(f"  Variance explained: {group_results['cumulative_variance']:.2f}%")
            print(f"  Component names: {', '.join(group_results['loadings'].columns)}")

        total_input = sum(len(r['feature_names']) for r in results.values())
        total_output = sum(r['n_components'] for r in results.values())

        print(f"\n{'='*80}")
        print(f"DIMENSIONALITY REDUCTION")
        print(f"{'='*80}")
        print(f"Total input features: {total_input}")
        print(f"Total output components: {total_output}")
        print(f"Reduction: {total_input - total_output} features ({(1 - total_output/total_input)*100:.1f}%)")

        # Note: Raw features (motion + temporal) are preserved
        raw_motion = len(self.metadata['raw_features']['motion'])
        raw_temporal = len(self.metadata['raw_features']['temporal'])

        print(f"\nPreserved raw features:")
        print(f"  Motion: {raw_motion} features")
        print(f"  Temporal: {raw_temporal} features")
        print(f"\nTotal features for training: {total_output + raw_motion + raw_temporal}")


def main():
    analyzer = PCAAnalyzer()

    # Run PCA analysis
    results = analyzer.run_pca_analysis()

    # Save results
    analyzer.save_results(results)

    # Print summary
    analyzer.print_summary(results)

    print("\n" + "="*80)
    print("✅ PCA ANALYSIS COMPLETE!")
    print("="*80)
    print("\nNext step: Build training GUI (Gradio)")


if __name__ == "__main__":
    main()
