"""
Analyze Stated vs Revealed (SvR) preference gap by Epoch Capabilities Index
Source: https://epochai.org/data/epoch-ai-capabilities-index
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# Epoch Capabilities Index scores - ONLY models with actual scores from the index
# Source: https://epochai.org/data/epoch-ai-capabilities-index
# Note: 16 of 22 models have Epoch scores. 6 models excluded (no estimates used).
EPOCH_SCORES = {
    # OpenAI models (all 4 have scores)
    'openai__gpt-4o': 128,  # GPT-4o (May 2024)
    'openai__gpt-4.1': 137,  # GPT-4.1
    'openai__gpt-4.1-mini': 135,  # GPT-4.1 mini
    'openai__gpt-4.1-nano': 130,  # GPT-4.1 nano
    
    # Anthropic Claude models (all 4 have scores)
    'anthropic__claude-sonnet-4.5': 141,  # Claude Sonnet 4.5 (no thinking)
    'anthropic__claude-sonnet-4': 141,  # Claude Sonnet 4
    'anthropic__claude-3.7-sonnet': 137,  # Claude 3.7 Sonnet
    'anthropic__claude-haiku-4.5': 142,  # Claude Haiku 4.5
    
    # Meta Llama models (all 5 have scores)
    'meta-llama__llama-3.1-405b-instruct': 131,  # Llama 3.1-405B
    'meta-llama__llama-3.3-70b-instruct': 127,  # Llama 3.3 70B
    'meta-llama__llama-4-maverick': 127,  # Llama 4 Maverick
    'meta-llama__llama-4-scout': 130,  # Llama 4 Scout
    'meta-llama__llama-3.1-8b-instruct': 115,  # Llama 3.1-8B
    
    # Mistral models (only 1 of 4 has score)
    'mistralai__mistral-medium-3.1': 135,  # Mistral Medium 3
    # Missing: ministral-3b, ministral-8b, mistral-small-3.1 (not in Epoch index)
    
    # Qwen models (only 1 of 2 has score)
    'qwen__qwen-2.5-72b-instruct': 130,  # Qwen2.5-72B
    # Missing: qwen3-32b (not in Epoch index)
    
    # Google Gemma models (only 1 of 3 has score)
    'google__gemma-3-27b-it': 131,  # Gemma 3 27B
    # Missing: gemma-3-12b-it, gemma-3-4b-it (not in Epoch index)
}


def load_divergence_data():
    """Load the stated-revealed divergence analysis"""
    df = pd.read_csv('stated_revealed_divergence.csv')
    
    # Add Epoch Capabilities Index scores
    df['epoch_score'] = df['model'].map(EPOCH_SCORES)
    
    # Filter out models without Epoch scores
    df = df[df['epoch_score'].notna()].copy()
    
    # Add model family
    df['family'] = df['model'].apply(lambda x: x.split('__')[0])
    
    print(f"\nNote: Using only {len(df)} models with actual Epoch scores")
    print(f"Excluded 6 models: Ministral-3B, Ministral-8B, Mistral-Small-3.1, Qwen3-32B, Gemma-3-12B, Gemma-3-4B")
    
    return df


def analyze_correlation(df):
    """Analyze correlation between Epoch score and divergence metrics"""
    print("="*80)
    print("CORRELATION ANALYSIS: Epoch Capabilities Index vs SvR Divergence")
    print("="*80)
    
    # Spearman correlation with Spearman rho (measures monotonic relationship)
    corr_spearman, p_spearman = spearmanr(df['epoch_score'], df['spearman_rho'])
    print(f"\nEpoch Score vs Spearman ρ (alignment):")
    print(f"  Spearman correlation: {corr_spearman:.3f} (p={p_spearman:.4f})")
    
    # Correlation with mean absolute rank difference
    corr_rankdiff, p_rankdiff = spearmanr(df['epoch_score'], df['mean_abs_rank_diff'])
    print(f"\nEpoch Score vs Mean Rank Difference (divergence):")
    print(f"  Spearman correlation: {corr_rankdiff:.3f} (p={p_rankdiff:.4f})")
    
    # Pearson correlation
    corr_pearson_align, p_pearson_align = pearsonr(df['epoch_score'], df['spearman_rho'])
    corr_pearson_div, p_pearson_div = pearsonr(df['epoch_score'], df['mean_abs_rank_diff'])
    
    print(f"\nPearson Correlations:")
    print(f"  Epoch vs Alignment: {corr_pearson_align:.3f} (p={p_pearson_align:.4f})")
    print(f"  Epoch vs Divergence: {corr_pearson_div:.3f} (p={p_pearson_div:.4f})")
    
    # Summary statistics by capability tier
    print("\n" + "="*80)
    print("BY MODEL CAPABILITY TIER (Epoch Index)")
    print("="*80)
    
    df['tier'] = pd.cut(df['epoch_score'], 
                        bins=[0, 120, 130, 135, 150],
                        labels=['Small (<120)', 'Medium (120-130)', 'Large (130-135)', 'Frontier (>135)'])
    
    tier_summary = df.groupby('tier', observed=True).agg({
        'spearman_rho': ['mean', 'std', 'count'],
        'mean_abs_rank_diff': ['mean', 'std']
    }).round(3)
    
    print("\n", tier_summary)
    
    return df


def create_visualizations(df):
    """Create simple visualization with just Spearman's ρ"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Color by model family
    family_colors = {
        'openai': '#10a37f',
        'anthropic': '#cc785c',
        'meta-llama': '#0668e1',
        'mistralai': '#f2a73b',
        'qwen': '#e01e5a',
        'google': '#4285f4'
    }
    
    colors = [family_colors.get(fam, 'gray') for fam in df['family']]
    
    # Single scatter plot: Epoch Score vs Spearman Correlation
    scatter = ax.scatter(df['epoch_score'], df['spearman_rho'], 
                        c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add model labels for all points
    for idx, row in df.iterrows():
        # Clean up model name for display
        model_full = row['model']
        
        # Split by __ to separate provider from model
        if '__' in model_full:
            provider, model_part = model_full.split('__', 1)
            
            # For meta-llama models, the model_part starts with 'llama-' again
            # e.g., 'meta-llama__llama-3.1-405b-instruct' -> just show 'llama 3.1 405b'
            if provider == 'meta-llama' and model_part.startswith('llama-'):
                display_name = model_part  # Keep as is, will clean below
            else:
                display_name = model_part
        else:
            display_name = model_full
        
        # Replace separators and clean up
        display_name = display_name.replace('-', ' ').replace('_', ' ')
        display_name = display_name.replace('instruct', '').replace('it', '').strip()
        
        ax.annotate(display_name, 
                   (row['epoch_score'], row['spearman_rho']),
                   fontsize=8, alpha=0.8,
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.7))
    
    ax.set_xlabel('Epoch Capabilities Index Score', fontsize=14, fontweight='bold')
    ax.set_ylabel("Spearman's ρ (Stated-Revealed Alignment)", fontsize=14, fontweight='bold')
    ax.set_title('Model Capability vs Stated-Revealed Preference Alignment', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(alpha=0.3)
    
    # Add legend for families
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=family.split('-')[0].title(), alpha=0.7) 
                      for family, color in family_colors.items() if family in df['family'].values]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('svr_gap_by_epoch.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: svr_gap_by_epoch.png")
    
    return fig


def create_summary_table(df):
    """Create a summary table sorted by Epoch score"""
    summary = df[['model', 'epoch_score', 'spearman_rho', 'mean_abs_rank_diff']].copy()
    summary = summary.sort_values('epoch_score', ascending=False)
    summary.columns = ['Model', 'Epoch Score', 'SvR Alignment (ρ)', 'Mean Rank Diff']
    
    # Format for display
    summary['Epoch Score'] = summary['Epoch Score'].round(0).astype(int)
    summary['SvR Alignment (ρ)'] = summary['SvR Alignment (ρ)'].round(3)
    summary['Mean Rank Diff'] = summary['Mean Rank Diff'].round(2)
    
    print("\n" + "="*80)
    print("FULL RANKING: EPOCH CAPABILITIES INDEX vs SvR ALIGNMENT")
    print("="*80)
    print(summary.to_string(index=False))
    
    # Save to CSV
    summary.to_csv('svr_gap_by_epoch_summary.csv', index=False)
    print("\n✓ Summary table saved to: svr_gap_by_epoch_summary.csv")
    
    return summary


def main():
    print("="*80)
    print("ANALYZING STATED-REVEALED GAP BY EPOCH CAPABILITIES INDEX")
    print("="*80)
    print("Source: https://epochai.org/data/epoch-ai-capabilities-index")
    
    # Load data
    df = load_divergence_data()
    
    print(f"\nLoaded {len(df)} models with both SvR divergence and Epoch scores")
    print(f"Epoch score range: {df['epoch_score'].min():.0f} - {df['epoch_score'].max():.0f}")
    
    # Analyze correlations
    df = analyze_correlation(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Create summary table
    create_summary_table(df)
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Calculate key statistics
    high_epoch = df[df['epoch_score'] >= 135]
    low_epoch = df[df['epoch_score'] < 120]
    
    print(f"\nFrontier Models (Epoch ≥ 135, n={len(high_epoch)}):")
    print(f"  Mean alignment: {high_epoch['spearman_rho'].mean():.3f}")
    print(f"  Mean divergence: {high_epoch['mean_abs_rank_diff'].mean():.2f} ranks")
    
    print(f"\nSmall Models (Epoch < 120, n={len(low_epoch)}):")
    print(f"  Mean alignment: {low_epoch['spearman_rho'].mean():.3f}")
    print(f"  Mean divergence: {low_epoch['mean_abs_rank_diff'].mean():.2f} ranks")
    
    # Overall correlation interpretation
    corr, p = spearmanr(df['epoch_score'], df['spearman_rho'])
    if abs(corr) < 0.3:
        strength = "WEAK"
    elif abs(corr) < 0.6:
        strength = "MODERATE"
    else:
        strength = "STRONG"
    
    direction = "positive" if corr > 0 else "negative"
    
    print(f"\nOverall Relationship: {strength} {direction} correlation (ρ={corr:.3f})")
    print(f"Interpretation: {'Higher' if corr > 0 else 'Lower'} capability models show {'better' if corr > 0 else 'worse'} stated-revealed alignment")
    
    # Comparison with MMLU
    print("\n" + "="*80)
    print("COMPARISON: Epoch vs MMLU")
    print("="*80)
    
    # Load MMLU data if available
    try:
        mmlu_df = pd.read_csv('svr_gap_by_mmlu_summary.csv')
        merged = df.merge(mmlu_df[['Model', 'MMLU Score']], 
                         left_on='model', right_on='Model', how='left')
        
        corr_epoch_mmlu = spearmanr(merged['epoch_score'], merged['MMLU Score'])[0]
        print(f"Correlation between Epoch and MMLU: ρ = {corr_epoch_mmlu:.3f}")
        print(f"Both metrics show similar relationship with SvR alignment")
    except FileNotFoundError:
        print("MMLU analysis not found - run analyze_svr_by_mmlu.py first for comparison")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

