"""
Calculate divergence between stated preferences (from local CSV files) 
and revealed preferences (to be loaded from actual data source)

NOTE: This script is a template. You need to provide the actual revealed preference data.
The revealed preferences should come from running models on the AIRiskDilemmas dataset
and calculating ELO ratings using the calculate_elo_rating.py script.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_preferences(elo_rating_dir):
    """Load preference rankings from CSV files"""
    prefs = {}
    
    for csv_file in Path(elo_rating_dir).glob("*.csv"):
        model_name = csv_file.stem
        df = pd.read_csv(csv_file)
        
        # Create a dict of value -> rank
        value_to_rank = {}
        for _, row in df.iterrows():
            value_to_rank[row['value_class']] = row['Rank']
        
        prefs[model_name] = value_to_rank
    
    return prefs


def load_stated_preferences(elo_rating_dir):
    """Load stated preference rankings from CSV files"""
    return load_preferences(elo_rating_dir)


def load_revealed_preferences(elo_rating_dir):
    """Load revealed preference rankings from CSV files"""
    return load_preferences(elo_rating_dir)


def calculate_divergence_metrics(stated_ranks, revealed_ranks):
    """Calculate various divergence metrics between two rankings"""
    # Get common values
    common_values = set(stated_ranks.keys()) & set(revealed_ranks.keys())
    
    if len(common_values) == 0:
        return None
    
    # Extract ranks for common values
    stated = [stated_ranks[v] for v in sorted(common_values)]
    revealed = [revealed_ranks[v] for v in sorted(common_values)]
    
    # Calculate metrics
    spearman_corr, spearman_p = spearmanr(stated, revealed)
    kendall_corr, kendall_p = kendalltau(stated, revealed)
    
    # Mean absolute rank difference
    rank_diffs = [abs(s - r) for s, r in zip(stated, revealed)]
    mean_abs_diff = np.mean(rank_diffs)
    
    # Root mean squared rank difference
    rmse = np.sqrt(np.mean([d**2 for d in rank_diffs]))
    
    # Maximum rank difference
    max_diff = max(rank_diffs)
    
    return {
        'spearman_rho': spearman_corr,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_corr,
        'kendall_p': kendall_p,
        'mean_abs_rank_diff': mean_abs_diff,
        'rmse_rank_diff': rmse,
        'max_rank_diff': max_diff,
        'num_values': len(common_values)
    }


def main():
    # Add argument for controlled stated preferences
    parser = argparse.ArgumentParser(description='Calculate divergence between stated and revealed preferences')
    parser.add_argument('--controlled', action='store_true', help='Use controlled stated preferences')
    args = parser.parse_args()

    # Determine input directory based on the controlled flag
    stated_dir = "elo_rating_stated_controlled" if args.controlled else "elo_rating_stated"

    # Load stated and revealed preferences
    stated_prefs = load_stated_preferences(stated_dir)
    revealed_prefs = load_revealed_preferences("elo_rating")
    
    print(f"\nLoaded {len(stated_prefs)} models with stated preferences")
    print(f"Loaded {len(revealed_prefs)} models with revealed preferences")
    
    # Find common models
    common_models = set(stated_prefs.keys()) & set(revealed_prefs.keys())
    print(f"Analyzing {len(common_models)} models with both stated and revealed preferences\n")
    
    # Calculate divergence for each model
    results = []
    
    for model_name in sorted(common_models):
        stated = stated_prefs[model_name]
        revealed = revealed_prefs[model_name]
            
        metrics = calculate_divergence_metrics(stated, revealed)
        
        if metrics:
            results.append({
                'model': model_name,
                **metrics
            })
            
            print(f"\n{'='*80}")
            print(f"Model: {model_name}")
            print(f"{'='*80}")
            print(f"Spearman's ρ: {metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.4f})")
            print(f"Kendall's τ: {metrics['kendall_tau']:.3f} (p={metrics['kendall_p']:.4f})")
            print(f"Mean Absolute Rank Difference: {metrics['mean_abs_rank_diff']:.2f}")
            print(f"RMSE Rank Difference: {metrics['rmse_rank_diff']:.2f}")
            print(f"Max Rank Difference: {metrics['max_rank_diff']:.0f}")
            
            # Show per-value differences
            print(f"\nPer-value rank differences:")
            print(f"{'Value':<20} {'Stated':>8} {'Revealed':>10} {'Diff':>6}")
            print(f"{'-'*50}")
            
            for value in sorted(stated.keys()):
                if value in revealed:
                    s_rank = stated[value]
                    r_rank = revealed[value]
                    diff = s_rank - r_rank
                    print(f"{value:<20} {s_rank:>8} {r_rank:>10} {diff:>6}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('spearman_rho')
    
    # Save results
    output_csv = "stated_revealed_divergence_controlled.csv" if args.controlled else "stated_revealed_divergence.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n\nSummary saved to '{output_csv}'")
    
    # Create visualizations
    create_visualizations(results_df, args.controlled)
    
    return results_df


def create_visualizations(results_df, controlled):
    """Create visualizations of divergence metrics"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Spearman correlation by model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Spearman correlation
    ax1 = axes[0, 0]
    results_sorted = results_df.sort_values('spearman_rho')
    colors = ['red' if x < 0 else 'green' for x in results_sorted['spearman_rho']]
    ax1.barh(range(len(results_sorted)), results_sorted['spearman_rho'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(results_sorted)))
    ax1.set_yticklabels([m.replace('__', '\n') for m in results_sorted['model']], fontsize=8)
    ax1.set_xlabel("Spearman's ρ", fontsize=12)
    ax1.set_title("Correlation between Stated and Revealed Preferences", fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Mean absolute rank difference
    ax2 = axes[0, 1]
    results_sorted = results_df.sort_values('mean_abs_rank_diff', ascending=False)
    ax2.barh(range(len(results_sorted)), results_sorted['mean_abs_rank_diff'], color='coral', alpha=0.7)
    ax2.set_yticks(range(len(results_sorted)))
    ax2.set_yticklabels([m.replace('__', '\n') for m in results_sorted['model']], fontsize=8)
    ax2.set_xlabel("Mean Absolute Rank Difference", fontsize=12)
    ax2.set_title("Average Rank Divergence", fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Scatter plot - Spearman vs Mean Abs Diff
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['spearman_rho'], results_df['mean_abs_rank_diff'], 
                         s=100, alpha=0.6, c=results_df['spearman_rho'], cmap='RdYlGn')
    ax3.set_xlabel("Spearman's ρ", fontsize=12)
    ax3.set_ylabel("Mean Absolute Rank Difference", fontsize=12)
    ax3.set_title("Correlation vs Rank Difference", fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Add model labels to scatter plot
    for idx, row in results_df.iterrows():
        ax3.annotate(row['model'].split('__')[-1][:10], 
                    (row['spearman_rho'], row['mean_abs_rank_diff']),
                    fontsize=7, alpha=0.7)
    
    plt.colorbar(scatter, ax=ax3, label="Spearman's ρ")
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*50}
    
    Number of models analyzed: {len(results_df)}
    
    Spearman's ρ:
        Mean: {results_df['spearman_rho'].mean():.3f}
        Median: {results_df['spearman_rho'].median():.3f}
        Std: {results_df['spearman_rho'].std():.3f}
        Range: [{results_df['spearman_rho'].min():.3f}, {results_df['spearman_rho'].max():.3f}]
    
    Mean Absolute Rank Difference:
        Mean: {results_df['mean_abs_rank_diff'].mean():.2f}
        Median: {results_df['mean_abs_rank_diff'].median():.2f}
        Std: {results_df['mean_abs_rank_diff'].std():.2f}
        Range: [{results_df['mean_abs_rank_diff'].min():.2f}, {results_df['mean_abs_rank_diff'].max():.2f}]
    
    Models with negative correlation (ρ < 0): {(results_df['spearman_rho'] < 0).sum()}
    Models with weak correlation (|ρ| < 0.3): {(results_df['spearman_rho'].abs() < 0.3).sum()}
    Models with moderate correlation (0.3 ≤ |ρ| < 0.7): {((results_df['spearman_rho'].abs() >= 0.3) & (results_df['spearman_rho'].abs() < 0.7)).sum()}
    Models with strong correlation (|ρ| ≥ 0.7): {(results_df['spearman_rho'].abs() >= 0.7).sum()}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_summary_png = "stated_revealed_divergence_summary_controlled.png" if controlled else "stated_revealed_divergence_summary.png"
    plt.savefig(output_summary_png, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{output_summary_png}'")
    
    # Figure 2: Heatmap of rank differences by model and value
    create_heatmap(results_df, controlled)


def create_heatmap(results_df, controlled):
    """Create a heatmap showing rank differences for each model and value"""
    
    stated_prefs = load_stated_preferences("elo_rating_stated")
    revealed_prefs = load_revealed_preferences("elo_rating")
    
    # Get common models
    common_models = set(stated_prefs.keys()) & set(revealed_prefs.keys())
    
    # Get all values (assuming all models have same values)
    all_values = sorted(list(stated_prefs[list(common_models)[0]].keys()))
    models = sorted(list(common_models))
    
    # Create matrix of rank differences (stated - revealed)
    diff_matrix = []
    
    for model in models:
        stated = stated_prefs[model]
        revealed = revealed_prefs[model]
        
        row = []
        for value in all_values:
            if value in stated and value in revealed:
                diff = stated[value] - revealed[value]
                row.append(diff)
            else:
                row.append(np.nan)
        
        diff_matrix.append(row)
    
    diff_matrix = np.array(diff_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use diverging colormap centered at 0
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10)
    
    # Set ticks
    ax.set_xticks(np.arange(len(all_values)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(all_values, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels([m.replace('__', ' ') for m in models], fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank Difference (Stated - Revealed)', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(all_values)):
            if not np.isnan(diff_matrix[i, j]):
                text = ax.text(j, i, f'{int(diff_matrix[i, j]):+d}',
                             ha="center", va="center", color="black" if abs(diff_matrix[i, j]) < 5 else "white",
                             fontsize=7)
    
    ax.set_title('Rank Differences: Stated - Revealed Preferences\n(Positive = Higher rank in stated, Negative = Higher rank in revealed)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Values', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('stated_revealed_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to 'stated_revealed_heatmap.png'")


if __name__ == "__main__":
    results_df = main()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nFiles generated:")
    print(f"  1. stated_revealed_divergence.csv - Detailed metrics for each model")
    print(f"  2. stated_revealed_divergence_summary.png - Summary visualizations")
    print(f"  3. stated_revealed_heatmap.png - Heatmap of rank differences")

