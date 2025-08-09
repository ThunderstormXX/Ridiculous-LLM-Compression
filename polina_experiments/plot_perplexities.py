#!/usr/bin/env python3
"""
Plot perplexity results across layers.

Usage:
    python plot_perplexities.py

This script:
1. Reads all perplexity results from logs/perplexities_layer*.json
2. Creates a plot showing perplexity changes across layers
3. Saves the plot and a summary CSV file
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Setup paths
LOGS_DIR = Path("logs")
FIGURES_DIR = Path("figures")

def load_perplexity_results():
    """Load all perplexity results from JSON files"""
    results = []
    
    # Find all perplexity result files
    pattern = str(LOGS_DIR / "perplexities_layer*.json")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No perplexity result files found in {LOGS_DIR}")
    
    print(f"Found {len(files)} result files")
    
    for file_path in sorted(files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def create_summary_dataframe(results):
    """Create a pandas DataFrame from results"""
    df_data = []
    
    for result in results:
        df_data.append({
            'layer': result['layer'],
            'before': result['before'],
            'after_similar': result['after_similar'],
            'after_dissimilar': result['after_dissimilar'],
            'similar_heads': str(result['similar_heads']),
            'dissimilar_heads': str(result['dissimilar_heads']),
            'max_similarity': result['max_similarity'],
            'min_similarity': result['min_similarity']
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('layer')
    return df

def plot_perplexities(df):
    """Create and save perplexity plot"""
    plt.figure(figsize=(12, 8))
    
    layers = df['layer']
    
    # Plot three lines
    plt.plot(layers, df['before'], 'b-o', label='Before Merging', linewidth=2, markersize=6)
    plt.plot(layers, df['after_similar'], 'g-s', label='After Merging Similar Heads', linewidth=2, markersize=6)
    plt.plot(layers, df['after_dissimilar'], 'r-^', label='After Merging Dissimilar Heads', linewidth=2, markersize=6)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Normalized Perplexity', fontsize=12)
    plt.title('Impact of Attention Head Merging on Perplexity', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save plot
    plot_path = FIGURES_DIR / "perplexity_vs_layer.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    plt.show()

def save_summary_csv(df):
    """Save summary data to CSV"""
    csv_path = LOGS_DIR / "perplexities_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

def print_statistics(df):
    """Print summary statistics"""
    print("\n=== Summary Statistics ===")
    print(f"Number of layers analyzed: {len(df)}")
    print(f"Average baseline perplexity: {df['before'].mean():.4f}")
    print(f"Average perplexity after similar head merging: {df['after_similar'].mean():.4f}")
    print(f"Average perplexity after dissimilar head merging: {df['after_dissimilar'].mean():.4f}")
    
    # Calculate changes
    similar_change = ((df['after_similar'] - df['before']) / df['before'] * 100).mean()
    dissimilar_change = ((df['after_dissimilar'] - df['before']) / df['before'] * 100).mean()
    
    print(f"\nAverage perplexity change:")
    print(f"  Similar heads merging: {similar_change:+.2f}%")
    print(f"  Dissimilar heads merging: {dissimilar_change:+.2f}%")
    
    # Find best and worst layers
    similar_changes = (df['after_similar'] - df['before']) / df['before'] * 100
    dissimilar_changes = (df['after_dissimilar'] - df['before']) / df['before'] * 100
    
    best_similar_layer = df.loc[similar_changes.idxmin(), 'layer']
    worst_similar_layer = df.loc[similar_changes.idxmax(), 'layer']
    best_dissimilar_layer = df.loc[dissimilar_changes.idxmin(), 'layer']
    worst_dissimilar_layer = df.loc[dissimilar_changes.idxmax(), 'layer']
    
    print(f"\nBest performing layers:")
    print(f"  Similar heads merging: Layer {best_similar_layer} ({similar_changes.min():+.2f}%)")
    print(f"  Dissimilar heads merging: Layer {best_dissimilar_layer} ({dissimilar_changes.min():+.2f}%)")
    
    print(f"\nWorst performing layers:")
    print(f"  Similar heads merging: Layer {worst_similar_layer} ({similar_changes.max():+.2f}%)")
    print(f"  Dissimilar heads merging: Layer {worst_dissimilar_layer} ({dissimilar_changes.max():+.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Plot perplexity results across layers")
    parser.add_argument("--no_plot", action="store_true", help="Skip showing the plot")
    args = parser.parse_args()
    
    print("Loading perplexity results...")
    
    # Load results
    results = load_perplexity_results()
    
    # Create DataFrame
    df = create_summary_dataframe(results)
    print(f"Loaded results for {len(df)} layers")
    
    # Print statistics
    print_statistics(df)
    
    # Create plot
    if not args.no_plot:
        print("\nCreating plot...")
        plot_perplexities(df)
    
    # Save CSV
    save_summary_csv(df)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()