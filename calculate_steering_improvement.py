import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_preferences(elo_rating_dir):
    """Load preference rankings from CSV files"""
    prefs = {}
    path = Path(elo_rating_dir)
    
    if not path.exists():
        print(f"Warning: Directory '{elo_rating_dir}' not found.")
        return {}

    # Look for all CSV files
    for csv_file in path.glob("*.csv"):
        # Use filename stem (e.g., 'openai__gpt-4o') as model ID
        model_name = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            
            # Check for required columns
            if 'value_class' not in df.columns or 'Rank' not in df.columns:
                print(f"Skipping {csv_file.name}: Missing 'value_class' or 'Rank' columns.")
                continue
                
            # Create a dict of value -> rank
            value_to_rank = {}
            for _, row in df.iterrows():
                value_to_rank[row['value_class']] = row['Rank']
            
            prefs[model_name] = value_to_rank
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
    
    return prefs

def calculate_correlation(rankings1, rankings2):
    """
    Calculate Spearman correlation between two sets of rankings.
    Returns (rho, p_value) or (None, None) if insufficient overlap.
    """
    # Find common values between the two sets
    common_values = set(rankings1.keys()) & set(rankings2.keys())
    
    # Need at least 2 points for correlation
    if len(common_values) < 2:
        return None, None
    
    # Extract ranks for common values, sorted alphabetically by value name to ensure alignment
    sorted_values = sorted(common_values)
    r1 = [rankings1[v] for v in sorted_values]
    r2 = [rankings2[v] for v in sorted_values]
    
    return spearmanr(r1, r2)

def main():
    parser = argparse.ArgumentParser(description='Calculate and plot improvement in alignment with stated preferences after steering.')
    parser.add_argument('--stated_dir', default="elo_rating_stated", help='Directory containing the target Stated Preferences')
    parser.add_argument('--revealed_dir', default="elo_rating", help='Directory containing the Baseline Revealed Preferences (Unsteered)')
    parser.add_argument('--steered_dir', default="elo_rating_steered_stated", help='Directory containing the Steered Revealed Preferences')
    parser.add_argument('--output_prefix', default="steering_analysis", help='Prefix for output CSV and plots')
    
    args = parser.parse_args()

    # 1. Load Data
    print(f"--- Loading Preferences ---")
    stated_prefs = load_preferences(args.stated_dir)
    revealed_prefs = load_preferences(args.revealed_dir)
    steered_prefs = load_preferences(args.steered_dir)
    
    print(f"Found {len(stated_prefs)} stated, {len(revealed_prefs)} baseline, and {len(steered_prefs)} steered models.")

    # 2. Identify Common Models
    common_models = set(stated_prefs.keys()) & set(revealed_prefs.keys()) & set(steered_prefs.keys())
    
    if not common_models:
        print("\nError: No common models found across all three directories.")
        print("Please check your file naming conventions (e.g. 'openai__gpt-4o.csv').")
        return

    print(f"Analyzing {len(common_models)} common models: {', '.join(sorted(common_models))}\n")

    # 3. Calculate Correlations
    results = []
    
    for model in sorted(common_models):
        target = stated_prefs[model]
        baseline = revealed_prefs[model]
        steered = steered_prefs[model]
        
        # Calc Baseline Correlation (Stated vs Baseline Revealed)
        rho_baseline, p_base = calculate_correlation(target, baseline)
        
        # Calc Steered Correlation (Stated vs Steered Revealed)
        rho_steered, p_steer = calculate_correlation(target, steered)
        
        if rho_baseline is not None and rho_steered is not None:
            improvement = rho_steered - rho_baseline
            
            results.append({
                'model': model,
                'rho_baseline': rho_baseline,
                'p_baseline': p_base,
                'rho_steered': rho_steered,
                'p_steered': p_steer,
                'improvement': improvement
            })
            
            print(f"Model: {model}")
            print(f"  Baseline Rho: {rho_baseline:.3f} (p={p_base:.3f})")
            print(f"  Steered Rho:  {rho_steered:.3f} (p={p_steer:.3f})")
            print(f"  Improvement:  {improvement:+.3f}")
            print("-" * 40)

    if not results:
        print("No valid results computed.")
        return

    # 4. Save Results to CSV
    df = pd.DataFrame(results)
    csv_path = f"{args.output_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed metrics saved to '{csv_path}'")

    # 5. Visualizations
    sns.set_style("whitegrid")
    
    # --- Plot A: Grouped Bar Chart (Side-by-Side Comparison) ---
    plt.figure(figsize=(12, 6))
    
    # Melt dataframe for seaborn
    df_melted = df.melt(id_vars=['model'], 
                        value_vars=['rho_baseline', 'rho_steered'], 
                        var_name='Condition', value_name='Correlation')
    
    # Rename for legend
    df_melted['Condition'] = df_melted['Condition'].map({
        'rho_baseline': 'Baseline (Unsteered)', 
        'rho_steered': 'Steered'
    })
    
    # Sort models by Improvement for better visual impact
    sorted_models = df.sort_values('improvement', ascending=False)['model'].tolist()
    clean_labels = [m.replace('__', '\n').replace('_', ' ') for m in sorted_models]
    
    ax = sns.barplot(
        data=df_melted, 
        x='model', 
        y='Correlation', 
        hue='Condition', 
        order=sorted_models,
        palette=['#95a5a6', '#2ecc71'] # Grey for baseline, Green for steered
    )
    
    plt.title(f"Effect of Steering on Alignment with Stated Preferences", fontsize=14, fontweight='bold')
    plt.ylabel("Spearman's Correlation (ρ)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(ticks=range(len(clean_labels)), labels=clean_labels, rotation=0)
    plt.legend(title=None)
    plt.ylim(-1.1, 1.1)
    
    plot_path_bar = f"{args.output_prefix}_barchart.png"
    plt.tight_layout()
    plt.savefig(plot_path_bar, dpi=300)
    print(f"Bar chart saved to '{plot_path_bar}'")

    # --- Plot B: Dumbbell Plot (Arrow Shift) ---
    # Good for visualizing the magnitude and direction of change
    plt.figure(figsize=(10, max(5, len(df)*0.8)))
    
    # Re-sort by baseline to show "where they started"
    df_shift = df.sort_values('rho_baseline', ascending=True)
    models = df_shift['model'].apply(lambda x: x.replace('__', ' ').replace('_', ' ')).tolist()
    y_pos = range(len(models))
    
    # Draw connecting lines
    plt.hlines(y=y_pos, xmin=df_shift['rho_baseline'], xmax=df_shift['rho_steered'], color='grey', alpha=0.5, linewidth=2)
    
    # Plot points
    plt.scatter(df_shift['rho_baseline'], y_pos, color='#95a5a6', label='Baseline', s=100, zorder=3)
    plt.scatter(df_shift['rho_steered'], y_pos, color='#2ecc71', label='Steered', s=100, zorder=3)
    
    # Add directional arrows for significant shifts
    for i, row in enumerate(df_shift.itertuples()):
        diff = row.rho_steered - row.rho_baseline
        if abs(diff) > 0.02:
            # Draw a small arrow in the middle of the line indicating direction
            mid = (row.rho_baseline + row.rho_steered) / 2
            dx = (row.rho_steered - row.rho_baseline) * 0.001 # tiny length just for head direction
            plt.arrow(mid, i, dx, 0, head_width=0.15, head_length=0.03, fc='black', ec='black', alpha=0.6)

    plt.yticks(y_pos, models, fontsize=11)
    plt.xlabel("Spearman's Correlation with Stated Preferences", fontsize=12)
    plt.title("Shift in Alignment: From Baseline to Steered", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.xlim(-1.1, 1.1)
    
    plot_path_shift = f"{args.output_prefix}_shift.png"
    plt.tight_layout()
    plt.savefig(plot_path_shift, dpi=300)
    print(f"Shift plot saved to '{plot_path_shift}'")

if __name__ == "__main__":
    main()