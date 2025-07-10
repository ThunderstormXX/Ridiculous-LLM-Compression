import torch
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from typing import Optional, List, Dict, Any
import warnings
import os
import gc
import logging
from datetime import datetime
import json

# UMAP for manifold learning
try:
    import umap
except ImportError:
    print("UMAP not found. Please run 'pip install umap-learn'. Manifold analysis will be skipped.")
    umap = None

# --- 1. CONFIGURATION & STYLING ---
# NOTE: This config is mainly for reference, the regeneration script loads existing data.
CONFIG = {
    "MODEL_NAME": "unsloth/Llama-3.2-3B-Instruct",
    "LAYERS_TO_SKIP_START": 3,
    "LAYERS_TO_SKIP_END": 1,
    "MANIFOLD_FEATURES": [
        "importance_score", "redundancy_score", "donor_impact",
        "recipient_impact", "symmetry_score", "neighborhood_cohesion"
    ]
}

# --- ENHANCED STYLING FOR PLOTS ---
sns.set_theme(style="whitegrid") # A slightly more modern theme
PLOT_CONFIG = {
    "dpi": 300,
    "title_fontsize": 20,
    "label_fontsize": 16,
    "tick_fontsize": 14,
    "legend_fontsize": 14,
    "linewidth": 2.5,
    "markersize": 8,
    "scatter_s": 60,
    "scatter_edgecolor": 'black',
    "scatter_linewidth": 0.5,
}

# --- UTILITY & HELPER FUNCTIONS ---
def setup_logging(log_dir):
    """Sets up logging to both file and console."""
    log_file = os.path.join(log_dir, 'plot_regeneration.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --- REFINED PLOTTING FUNCTIONS ---

def _plot_perplexity_heatmap(results_matrix, layers_to_skip, save_path=None, ax=None):
    if ax is None: fig, ax_local = plt.subplots(figsize=(14, 12))
    else: ax_local = ax; fig = ax.get_figure()

    num_layers = results_matrix.shape[0]
    # Capping the perplexity for better color contrast
    vmax = np.nanpercentile(results_matrix, 99) # Use 99th percentile for robust scaling
    vmin = np.nanmin(results_matrix)
    sns.heatmap(results_matrix, ax=ax_local, cmap="viridis_r", annot=False, vmin=vmin, vmax=vmax, cbar=ax is None)

    ax_local.set_title('Functional Similarity (Perplexity upon Replacement)', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
    ax_local.set_ylabel('Target Layer (Replaced)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.set_xlabel('Source Layer (Copied)', fontsize=PLOT_CONFIG['label_fontsize'])

    for layer_idx in layers_to_skip:
        if layer_idx < num_layers:
            ax_local.add_patch(plt.Rectangle((0, layer_idx), num_layers, 1, fill=True, color='gray', alpha=0.5, zorder=10))
            ax_local.add_patch(plt.Rectangle((layer_idx, 0), 1, num_layers, fill=True, color='gray', alpha=0.5, zorder=10))
    ax_local.plot([0, num_layers], [0, num_layers], color='red', linestyle=':', linewidth=2.5)
    sns.despine(ax=ax_local, trim=True)

    if ax is None:
        cbar = ax_local.collections[0].colorbar
        cbar.set_label(f'Perplexity (Capped at {vmax:.1f})', rotation=270, labelpad=20, fontsize=PLOT_CONFIG['label_fontsize'])
        fig.tight_layout(); plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight'); plt.close(fig)
        logging.info(f"Perplexity heatmap saved to {save_path}")

def _plot_pruning_decay_curves(history_remove, history_replace, save_path=None, ax=None):
    if ax is None: fig, ax_local = plt.subplots(figsize=(12, 8))
    else: ax_local = ax; fig = ax.get_figure()

    # --- FIX: Set a dynamic Y-limit to prevent outliers from squashing the plot ---
    all_ppls = [h['ppl'] for h in history_remove] + [h['ppl'] for h in history_replace]
    all_ppls_finite = [p for p in all_ppls if np.isfinite(p)]
    if all_ppls_finite:
        baseline_ppl = history_remove[0]['ppl']
        upper_bound = np.percentile(all_ppls_finite, 98) # Show 98% of data comfortably
        ax_local.set_ylim(bottom=baseline_ppl * 0.9, top=upper_bound * 1.1)

    ax_local.plot([h['layers_remaining'] for h in history_remove], [h['ppl'] for h in history_remove], marker='o', linestyle='-', label='Pruning by Removal', linewidth=PLOT_CONFIG['linewidth'], markersize=PLOT_CONFIG['markersize'])
    ax_local.plot([h['layers_remaining'] for h in history_replace], [h['ppl'] for h in history_replace], marker='x', linestyle='--', label='Pruning by Replacement', linewidth=PLOT_CONFIG['linewidth'], markersize=PLOT_CONFIG['markersize'])
    
    # Add baseline for context
    if all_ppls_finite:
        ax_local.axhline(y=baseline_ppl, color='red', linestyle=':', linewidth=2, label=f'Baseline PPL ({baseline_ppl:.2f})')
        
    ax_local.set_title('Model Performance vs. Number of Layers', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
    ax_local.set_xlabel('Number of Remaining Editable Layers', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.set_ylabel('Perplexity (Lower is Better)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.invert_xaxis()
    ax_local.legend(fontsize=PLOT_CONFIG['legend_fontsize'])
    ax_local.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['tick_fontsize'])
    sns.despine(ax=ax_local)

    if ax is None:
        fig.tight_layout(); plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight'); plt.close(fig)
        logging.info(f"Pruning decay plot saved to {save_path}")

def _plot_cka_heatmap(cka_matrix, layers_to_skip, save_path=None, ax=None):
    if ax is None: fig, ax_local = plt.subplots(figsize=(14, 12))
    else: ax_local = ax; fig = ax.get_figure()

    num_layers = cka_matrix.shape[0]
    
    # --- FIX: Set vmax based on OFF-DIAGONAL values for better contrast ---
    off_diagonal_matrix = cka_matrix.copy()
    np.fill_diagonal(off_diagonal_matrix, np.nan)
    vmax = np.nanpercentile(off_diagonal_matrix, 99.5) # Almost the max off-diagonal value
    
    sns.heatmap(cka_matrix, ax=ax_local, cmap="magma", annot=False, vmin=0, vmax=vmax, cbar=ax is None)
    ax_local.set_title('Structural Similarity (CKA of Weights)', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
    
    for layer_idx in layers_to_skip:
        if layer_idx < num_layers:
            ax_local.add_patch(plt.Rectangle((0, layer_idx), num_layers, 1, fill=True, color='gray', alpha=0.5, zorder=10))
            ax_local.add_patch(plt.Rectangle((layer_idx, 0), 1, num_layers, fill=True, color='gray', alpha=0.5, zorder=10))
    ax_local.plot([0, num_layers], [0, num_layers], color='cyan', linestyle=':', linewidth=2.5)
    sns.despine(ax=ax_local, trim=True)

    if ax is None:
        cbar = ax_local.collections[0].colorbar
        cbar.set_label(f'CKA Similarity (Capped at {vmax:.2f})', rotation=270, labelpad=20, fontsize=PLOT_CONFIG['label_fontsize'])
        fig.tight_layout(); plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight'); plt.close(fig)
        logging.info(f"CKA heatmap saved to {save_path}")

def _plot_shockwave(results, save_path=None, ax=None):
    if ax is None: fig, ax_local = plt.subplots(figsize=(12, 8))
    else: ax_local = ax; fig = ax.get_figure()

    colors = sns.color_palette("plasma", n_colors=len(results))
    for i, (key, divergence) in enumerate(results.items()):
        label = key.replace("_", " ").replace("target", "T=").replace("source", " S=")
        ax_local.plot(range(len(divergence)), divergence, marker='.', linestyle='-', label=label, alpha=0.9, color=colors[i], linewidth=PLOT_CONFIG['linewidth'])
    if results:
        target_layer = int(list(results.keys())[0].split('_')[0].replace('target',''))
        ax_local.axvline(x=target_layer + 1, color='red', linestyle='--', label=f'Perturbation at Layer {target_layer}')
    
    # --- FIX: Tighten the Y-axis to the actual data range ---
    all_divergence = [val for div_list in results.values() for val in div_list]
    if all_divergence:
        ax_local.set_ylim(0, max(all_divergence) * 1.1)

    ax_local.set_title('Representational "Shockwave"', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
    ax_local.set_xlabel('Layer Index (0=Embeddings)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.set_ylabel('Divergence (1 - CKA)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.legend(fontsize=PLOT_CONFIG['legend_fontsize'])
    ax_local.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['tick_fontsize'])
    sns.despine(ax=ax_local)

    if ax is None:
        fig.tight_layout(); plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight'); plt.close(fig)
        logging.info(f"Shockwave plot saved to {save_path}")

def _plot_parameter_averaging(results, save_path=None, ax=None):
    if not results: return
    if ax is None: fig, ax_local = plt.subplots(figsize=(12, 8))
    else: ax_local = ax; fig = ax.get_figure()

    colors = sns.color_palette("rocket_r", n_colors=len(results))
    all_ppls = [v for data in results.values() for v in data.values()]
    
    for i, (pair_key, data) in enumerate(results.items()):
        items = sorted([(float(k), v) for k, v in data.items()])
        alphas_plot, ppls_plot = [item[0] for item in items], [item[1] for item in items]
        i_layer, j_layer = map(int, pair_key.split('->'))
        ax_local.plot(alphas_plot, ppls_plot, marker='o', linestyle='-', label=f"Avg({i_layer}, {j_layer})", color=colors[i], linewidth=PLOT_CONFIG['linewidth'], markersize=PLOT_CONFIG['markersize'])
    
    # --- FIX: Set Y-limit to focus on the more interesting, lower-PPL curves ---
    if all_ppls:
        min_ppl = np.min(all_ppls)
        upper_bound = np.percentile(all_ppls, 90) # Focus on the best 90% of results
        ax_local.set_ylim(min_ppl * 0.95, upper_bound * 1.1)
    
    ax_local.set_title('Perplexity during Layer Parameter Interpolation', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
    ax_local.set_xlabel('Alpha (0 = Original Layer, 1 = Source Layer)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.set_ylabel('Perplexity (Lower is Better)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.legend(fontsize=PLOT_CONFIG['legend_fontsize'], title="Replaced Target with:")
    ax_local.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['tick_fontsize'])
    sns.despine(ax=ax_local)
    
    if ax is None:
        fig.tight_layout(); plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight'); plt.close(fig)
        logging.info(f"Parameter averaging plot saved to {save_path}")

def _plot_manifold(feature_df, layers_to_skip, save_path=None, ax=None):
    if umap is None or feature_df is None or feature_df.empty: return
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42, n_components=2)
    
    valid_layers_df = feature_df.drop(index=list(layers_to_skip), errors='ignore')
    if valid_layers_df.empty:
        logging.warning("No valid layers found for UMAP analysis.")
        return
        
    for col in CONFIG["MANIFOLD_FEATURES"]:
        if col not in valid_layers_df.columns: valid_layers_df[col] = 0
            
    embedding = reducer.fit_transform(valid_layers_df[CONFIG["MANIFOLD_FEATURES"]].fillna(0))
    valid_layers_df['umap1'], valid_layers_df['umap2'] = embedding[:, 0], embedding[:, 1]
    
    if ax is None: fig, ax_local = plt.subplots(figsize=(15, 12))
    else: ax_local = ax; fig = ax.get_figure()

    # --- FIX: Make circles bigger ---
    sizes = valid_layers_df['importance_score']
    sizes_norm = 100 + 1000 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)
    
    scatter = ax_local.scatter(
        valid_layers_df['umap1'], valid_layers_df['umap2'], 
        s=sizes_norm.fillna(100), c=valid_layers_df.index, cmap='plasma', 
        alpha=0.85, 
        edgecolors=PLOT_CONFIG['scatter_edgecolor'], 
        linewidth=PLOT_CONFIG['scatter_linewidth']
    )
    
    for i in valid_layers_df.index:
        ax_local.text(valid_layers_df.loc[i, 'umap1'], valid_layers_df.loc[i, 'umap2'] + 0.05, str(i), ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax_local.set_title(f'UMAP Manifold of Layer Functional Signatures', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
    ax_local.set_xlabel('UMAP Dimension 1', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.set_ylabel('UMAP Dimension 2', fontsize=PLOT_CONFIG['label_fontsize'])
    ax_local.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['tick_fontsize'])
    sns.despine(ax=ax_local)
    
    if ax is None:
        cbar = plt.colorbar(scatter, ax=ax_local)
        cbar.set_label('Layer Index (Depth)', fontsize=PLOT_CONFIG['label_fontsize'])
        legend_elements = [
            plt.scatter([], [], s=100 + 1000 * val, label=f'{label} Importance')
            for val, label in [(0, 'Low'), (0.5, 'Mid'), (1, 'High')]
        ]
        ax_local.legend(handles=legend_elements, title="Size ~ Importance", fontsize=PLOT_CONFIG['legend_fontsize'], loc='best')
        fig.tight_layout(); plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight'); plt.close(fig)
        logging.info(f"Manifold plot saved to {save_path}")

def plot_correlation(ax, perplexity_matrix, cka_matrix, layers_to_skip):
    num_layers = cka_matrix.shape[0]
    cka_scores, ppl_scores, depths = [], [], []
    for i in range(num_layers):
        if i in layers_to_skip: continue
        for j in range(num_layers):
            if j in layers_to_skip or i == j: continue
            if not np.isnan(cka_matrix[i, j]) and not np.isnan(perplexity_matrix[i, j]):
                cka_scores.append(cka_matrix[i, j])
                ppl_scores.append(perplexity_matrix[i, j])
                depths.append(i)
    if not cka_scores: return
    
    correlation = pd.Series(cka_scores).corr(pd.Series(ppl_scores))
    
    # --- FIX: Make scatter points bigger ---
    scatter = ax.scatter(cka_scores, ppl_scores, c=depths, cmap='plasma', alpha=0.7, 
                         s=PLOT_CONFIG['scatter_s'], 
                         edgecolors=PLOT_CONFIG['scatter_edgecolor'], 
                         linewidth=PLOT_CONFIG['scatter_linewidth'])
                         
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Target Layer Depth', fontsize=PLOT_CONFIG['label_fontsize'])
    
    ax.set_title(f'Structural vs. Functional Similarity\n(Correlation: {correlation:.2f})', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
    ax.set_xlabel('CKA Similarity (Structure)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax.set_ylabel('Replacement Perplexity (Function)', fontsize=PLOT_CONFIG['label_fontsize'])
    ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['tick_fontsize'])
    
    # Set a reasonable Y limit to avoid outliers
    ppl_array = np.array(ppl_scores)
    y_upper = np.percentile(ppl_array[np.isfinite(ppl_array)], 99)
    ax.set_ylim(bottom=np.min(ppl_array) * 0.9, top=y_upper * 1.05)
    sns.despine(ax=ax)

# --- NEW/MODIFIED ORCHESTRATORS ---

def _plot_correlation_standalone(perplexity_matrix, cka_matrix, layers_to_skip, save_path):
    """Creates a standalone plot for the correlation analysis."""
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_correlation(ax, perplexity_matrix, cka_matrix, layers_to_skip)
    fig.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Standalone correlation plot saved to {save_path}")

def create_dashboard_plot(paths: Dict[str, str], results: Dict[str, Any], layers_to_skip: set):
    logging.info("\n--- Creating Final Analysis Dashboard with Enhanced Visuals ---")
    fig = plt.figure(figsize=(26, 32), constrained_layout=True)
    
    model_name_simple = paths.get('exp_name', 'Unknown Model')
    fig.suptitle(f"Comprehensive Layer Analysis for {model_name_simple}", fontsize=32, weight='bold')

    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1, 1, 1.5], hspace=0.45, wspace=0.25)

    def plot_placeholder(ax, title, msg="Data Not Found"):
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=18, color='gray', style='italic', weight='bold')
        ax.set_title(title, fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
        ax.set_xticks([]); ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

    # Row 1: Heatmaps
    ax_ppl = fig.add_subplot(gs[0, 0])
    if "perplexity_matrix" in results: _plot_perplexity_heatmap(results["perplexity_matrix"], layers_to_skip, ax=ax_ppl)
    else: plot_placeholder(ax_ppl, 'Functional Similarity (Perplexity)')
    
    ax_cka = fig.add_subplot(gs[0, 1])
    if "cka_matrix" in results: _plot_cka_heatmap(results["cka_matrix"], layers_to_skip, ax=ax_cka)
    else: plot_placeholder(ax_cka, 'Structural Similarity (CKA)')

    # Row 2: Pruning & Memory (No changes requested, but benefits from new style)
    ax_pruning = fig.add_subplot(gs[1, 0])
    if "history_remove" in results: _plot_pruning_decay_curves(results["history_remove"], results["history_replace"], ax=ax_pruning)
    else: plot_placeholder(ax_pruning, 'Model Performance vs. Number of Layers')
    
    ax_vram = fig.add_subplot(gs[1, 1])
    # Assuming the original plotting function for VRAM is sufficient
    if "history_remove" in results:
        ax_vram.plot([h['layers_remaining'] for h in results["history_remove"]], [h.get('vram_gb', 0) for h in results["history_remove"]], marker='o', linestyle='-', label='VRAM usage (Removal)', linewidth=PLOT_CONFIG['linewidth'])
        ax_vram.plot([h['layers_remaining'] for h in results["history_replace"]], [h.get('vram_gb', 0) for h in results["history_replace"]], marker='x', linestyle='--', label='VRAM usage (Replacement)', linewidth=PLOT_CONFIG['linewidth'])
        ax_vram.set_title('VRAM Usage During Pruning', fontsize=PLOT_CONFIG['title_fontsize'], pad=20)
        ax_vram.set_xlabel('Number of Remaining Editable Layers', fontsize=PLOT_CONFIG['label_fontsize'])
        ax_vram.set_ylabel('Allocated VRAM (GB)', fontsize=PLOT_CONFIG['label_fontsize'])
        ax_vram.invert_xaxis(); ax_vram.legend(fontsize=PLOT_CONFIG['legend_fontsize'])
        sns.despine(ax=ax_vram)
    else: plot_placeholder(ax_vram, 'VRAM Usage During Pruning')

    # Row 3: Shockwave & Interpolation
    ax_shockwave = fig.add_subplot(gs[2, 0])
    if "shockwave" in results: _plot_shockwave(results["shockwave"], ax=ax_shockwave)
    else: plot_placeholder(ax_shockwave, 'Representational "Shockwave"')

    ax_param_avg = fig.add_subplot(gs[2, 1])
    if "param_avg" in results: _plot_parameter_averaging(results["param_avg"], ax=ax_param_avg)
    else: plot_placeholder(ax_param_avg, 'Layer Parameter Interpolation')
    
    # Row 4: Manifold and Correlation
    ax_manifold = fig.add_subplot(gs[3, 0])
    if "manifold_df" in results and results["manifold_df"] is not None: _plot_manifold(results["manifold_df"], layers_to_skip, ax=ax_manifold)
    else: plot_placeholder(ax_manifold, 'UMAP Manifold of Layer Signatures')

    ax_corr = fig.add_subplot(gs[3, 1])
    if "perplexity_matrix" in results and "cka_matrix" in results:
        plot_correlation(ax_corr, results["perplexity_matrix"], results["cka_matrix"], layers_to_skip)
    else:
        plot_placeholder(ax_corr, 'Structural vs. Functional Similarity')

    plt.savefig(paths["plot_dashboard"], dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    logging.info(f"SUCCESS: Dashboard plot regenerated and saved to {paths['plot_dashboard']}")


def main_regenerate_plots(exp_dir: str):
    """
    Loads existing experiment data and regenerates all plots with improved visuals.
    """
    data_dir = os.path.join(exp_dir, "data")
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    setup_logging(os.path.join(exp_dir, "logs"))

    exp_name = os.path.basename(exp_dir).split('_202')[0].replace('_', '/')
    
    paths = {
        "exp_name": exp_name,
        "perplexity_matrix": os.path.join(data_dir, "perplexity_matrix.npy"),
        "cka_matrix": os.path.join(data_dir, "cka_matrix.npy"),
        "pruning_history_remove": os.path.join(data_dir, "pruning_history_remove.json"),
        "pruning_history_replace": os.path.join(data_dir, "pruning_history_replace.json"),
        "shockwave_results": os.path.join(data_dir, "shockwave_results.json"),
        "param_avg_results": os.path.join(data_dir, "param_avg_results.json"),
        "manifold_features": os.path.join(data_dir, "manifold_features.csv"),
        
        "plot_perplexity_heatmap": os.path.join(plots_dir, "1_perplexity_heatmap_revised.png"),
        "plot_pruning_decay": os.path.join(plots_dir, "2_pruning_decay_revised.png"),
        "plot_cka_heatmap": os.path.join(plots_dir, "4_cka_heatmap_revised.png"),
        "plot_shockwave": os.path.join(plots_dir, "5_shockwave_analysis_revised.png"),
        "plot_param_averaging": os.path.join(plots_dir, "6_parameter_averaging_revised.png"),
        "plot_manifold_viz": os.path.join(plots_dir, "7_manifold_visualization_revised.png"),
        "plot_correlation_standalone": os.path.join(plots_dir, "8_correlation_standalone_revised.png"),
        "plot_dashboard": os.path.join(exp_dir, f"__dashboard_{os.path.basename(exp_dir)}_REVISED.png"),
    }

    # Load all existing data
    results = {}
    for key, path in paths.items():
        if not os.path.exists(path) or "plot" in key: continue
        logging.info(f"Loading data from: {path}")
        try:
            if path.endswith(".npy"):
                results[key] = np.load(path)
            elif path.endswith(".json"):
                with open(path, 'r') as f:
                    results[key] = json.load(f)
            elif path.endswith(".csv"):
                results[key] = pd.read_csv(path, index_col=0)
        except Exception as e:
            logging.error(f"Failed to load {path}: {e}")

    # Determine layers to skip (approximated from data)
    num_layers = results.get("perplexity_matrix", np.array([[]])).shape[0]
    if num_layers == 0:
        logging.critical("Could not determine number of layers from perplexity_matrix. Aborting.")
        return
        
    # Reconstruct layers_to_skip from the NaN values in the perplexity matrix
    ppl_matrix = results.get("perplexity_matrix")
    layers_to_skip = set(np.where(np.isnan(ppl_matrix).all(axis=1))[0])
    logging.info(f"Inferred {num_layers} layers, with skipped layers: {sorted(list(layers_to_skip))}")


    # Regenerate standalone plots
    if "perplexity_matrix" in results:
        _plot_perplexity_heatmap(results["perplexity_matrix"], layers_to_skip, save_path=paths["plot_perplexity_heatmap"])
    if "history_remove" in results:
         _plot_pruning_decay_curves(results["history_remove"], results["history_replace"], save_path=paths["plot_pruning_decay"])
    if "cka_matrix" in results:
        _plot_cka_heatmap(results["cka_matrix"], layers_to_skip, save_path=paths["plot_cka_heatmap"])
    if "shockwave_results" in results:
        _plot_shockwave(results["shockwave_results"], save_path=paths["plot_shockwave"])
    if "param_avg_results" in results:
        _plot_parameter_averaging(results["param_avg_results"], save_path=paths["plot_param_averaging"])
    if "manifold_features" in results:
        _plot_manifold(results["manifold_features"], layers_to_skip, save_path=paths["plot_manifold_viz"])
    if "perplexity_matrix" in results and "cka_matrix" in results:
        _plot_correlation_standalone(results["perplexity_matrix"], results["cka_matrix"], layers_to_skip, save_path=paths["plot_correlation_standalone"])

    # Rename keys for dashboard function
    dashboard_results = {
        'perplexity_matrix': results.get('perplexity_matrix'),
        'cka_matrix': results.get('cka_matrix'),
        'history_remove': results.get('pruning_history_remove'),
        'history_replace': results.get('pruning_history_replace'),
        'shockwave': results.get('shockwave_results'),
        'param_avg': results.get('param_avg_results'),
        'manifold_df': results.get('manifold_features'),
    }

    # Regenerate the main dashboard
    create_dashboard_plot(paths, dashboard_results, layers_to_skip)

if __name__ == '__main__':
    # --- HOW TO RUN ---
    # 1. Save this script as `regenerate_plots.py`.
    # 2. Open your terminal or command prompt.
    # 3. Run the script and pass the path to your experiment directory.
    #
    # EXAMPLE:
    # python regenerate_plots.py "experiments/unsloth_Llama-3.2-3B-Instruct_2025-07-09_15-42"
    
    import sys
    if len(sys.argv) != 2:
        print("\nUsage: python regenerate_plots.py <path_to_experiment_directory>")
        print("Example: python regenerate_plots.py 'experiments/unsloth_Llama-3.2-3B-Instruct_2025-07-09_15-42'")
    else:
        experiment_directory = sys.argv[1]
        if not os.path.isdir(experiment_directory):
            print(f"\nError: Directory not found at '{experiment_directory}'")
        else:
            main_regenerate_plots(experiment_directory)
