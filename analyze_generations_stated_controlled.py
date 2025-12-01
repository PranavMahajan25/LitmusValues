import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

# Configuration
INPUT_FOLDER = "generations_stated_controlled"
OUTPUT_FILENAME = "stated_controlled_summary_plot.png"

# New Configuration for Correlation Analysis
ELO_STATED_DIR = "elo_rating_stated"
ELO_CONTROLLED_DIR = "elo_rating_stated_controlled"
CORRELATION_OUTPUT_FILENAME = "stated_vs_controlled_correlation.png"

def load_data(folder):
    """Load and aggregate all .summary.csv files."""
    files = glob.glob(os.path.join(folder, "*.summary.csv"))
    
    if not files:
        print(f"No summary files found in {folder}")
        return pd.DataFrame()
        
    data = []
    for f in files:
        try:
            # Filename format: model_name.summary.csv
            base_name = os.path.basename(f).replace(".summary.csv", "")
            
            # Read CSV: columns [category, count, percentage]
            df = pd.read_csv(f)
            
            # Convert to dictionary: {'binary': 3.5, 'equal': 0.0, 'depends': 96.5}
            # Normalize keys to lowercase just in case
            row_dict = dict(zip(df['category'].str.lower(), df['percentage']))
            
            record = {
                'model': base_name,
                'depends': row_dict.get('depends', 0),
                'equal': row_dict.get('equal', 0),
                'binary': row_dict.get('binary', 0)
            }
            data.append(record)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame(data)

def plot_summary(df):
    """Generate the stacked horizontal bar plot (Binary, Equal, Depends)."""
    if df.empty:
        return

    # Sort by 'binary' ascending so the highest 'binary' is at the top of the barh plot
    df = df.sort_values('binary', ascending=True)
    
    # Set the plotting style
    sns.set_style("whitegrid")
    
    # Calculate figure height based on number of models
    height = max(6, len(df) * 0.5)
    fig, ax = plt.subplots(figsize=(12, height))
    
    # Stack order: Binary, Equal, Depends
    plot_df = df[['binary', 'equal', 'depends']].copy()
    
    # Clean model names for Y-axis labels
    plot_df.index = df['model'].apply(lambda x: x.replace("__", "\n"))
    
    # Colors: Binary (Red), Equal (Blue), Depends (Grey)
    colors = ['#e74c3c', '#3498db', '#95a5a6']
    
    # Create Stacked Horizontal Bar Plot
    plot_df.plot(kind='barh', stacked=True, ax=ax, color=colors, width=0.8, edgecolor='white')
    
    # Labels and Titles
    ax.set_title('Distribution of Response Categories (Stated - Controlled)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlim(0, 100)
    
    # Format Y-tick labels
    ax.tick_params(axis='y', labelsize=9)
    
    # Add Legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [l.capitalize() for l in labels]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    
    # Manually add percentage text labels
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        
        # Only label if segment is wide enough (> 4%)
        if width > 4:
            ax.text(x + width/2, 
                    y + height/2, 
                    f'{width:.1f}%', 
                    ha='center', 
                    va='center',
                    fontsize=8,
                    color='white',
                    fontweight='bold')
    
    # Save path to current directory
    output_path = os.path.join('./', OUTPUT_FILENAME)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

# --- New Section: Correlation Analysis ---

def load_preferences(elo_dir):
    """Load preference rankings from CSV files in the directory."""
    prefs = {}
    path = Path(elo_dir)
    
    if not path.exists():
        print(f"Warning: Directory '{elo_dir}' not found.")
        return prefs

    for csv_file in path.glob("*.csv"):
        model_name = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            # Create a dict of value_class -> Rank
            # Assuming columns 'value_class' and 'Rank' exist as per your reference
            if 'value_class' in df.columns and 'Rank' in df.columns:
                value_to_rank = {}
                for _, row in df.iterrows():
                    value_to_rank[row['value_class']] = row['Rank']
                prefs[model_name] = value_to_rank
            else:
                print(f"Skipping {csv_file.name}: Missing 'value_class' or 'Rank' columns.")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return prefs

def analyze_correlations():
    """Compute and plot Spearman correlation between Stated and Controlled Stated preferences."""
    print("\n--- Starting Correlation Analysis ---")
    
    stated_prefs = load_preferences(ELO_STATED_DIR)
    controlled_prefs = load_preferences(ELO_CONTROLLED_DIR)
    
    print(f"Loaded {len(stated_prefs)} stated models.")
    print(f"Loaded {len(controlled_prefs)} controlled models.")
    
    # Find common models
    common_models = set(stated_prefs.keys()) & set(controlled_prefs.keys())
    if not common_models:
        print("No common models found between the two directories.")
        return

    results = []
    for model in common_models:
        s_ranks = stated_prefs[model]
        c_ranks = controlled_prefs[model]
        
        # Get common values to compare
        common_values = set(s_ranks.keys()) & set(c_ranks.keys())
        if len(common_values) < 2:
            continue
            
        # Create vectors for correlation
        sorted_values = sorted(common_values)
        s_vec = [s_ranks[v] for v in sorted_values]
        c_vec = [c_ranks[v] for v in sorted_values]
        
        rho, _ = spearmanr(s_vec, c_vec)
        results.append({'model': model, 'spearman_rho': rho})
    
    if not results:
        print("Not enough data to compute correlations.")
        return
        
    df_corr = pd.DataFrame(results)
    
    # Sort by Spearman Rho Ascending (so highest is at the top of the bar chart)
    df_corr = df_corr.sort_values('spearman_rho', ascending=True)
    
    # Plotting
    plot_correlations(df_corr)

def plot_correlations(df):
    """Plot horizontal bar chart of Spearman correlations."""
    sns.set_style("whitegrid")
    
    height = max(6, len(df) * 0.5)
    fig, ax = plt.subplots(figsize=(10, height))
    
    # Color scheme: Red if negative, Green if positive (from reference script)
    colors = ['red' if x < 0 else 'green' for x in df['spearman_rho']]
    
    # Plot horizontal bars
    bars = ax.barh(range(len(df)), df['spearman_rho'], color=colors, alpha=0.7)
    
    # Set labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([m.replace('__', '\n') for m in df['model']], fontsize=9)
    
    ax.set_xlabel("Spearman's ρ", fontsize=12)
    ax.set_title("Correlation: Stated vs. Stated Controlled", fontsize=14, fontweight='bold')
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(df['spearman_rho']):
        # Position label slightly offset from the bar end
        offset = 0.05 if v >= 0 else -0.05
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.2f}', va='center', ha=ha, fontsize=8)
    
    # Save
    output_path = os.path.join('./', CORRELATION_OUTPUT_FILENAME)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved to {output_path}")

if __name__ == "__main__":
    # Part 1: Stated Controlled Summary (Response Categories)
    print("Loading summary files for stacked bar plot...")
    df_results = load_data(INPUT_FOLDER)
    
    if not df_results.empty:
        print(f"Loaded data for {len(df_results)} models.")
        plot_summary(df_results)
    else:
        print("No summary data found.")
        
    # Part 2: Correlation Analysis (Stated vs Controlled)
    analyze_correlations()