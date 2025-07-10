import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
import gc

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore")

def evaluate_perplexity(
    model, tokenizer, device, num_texts=5, max_length=1024, stride=512
):
    """
    Calculates the perplexity of a Hugging Face model on a subset of the wikitext-2 test set.
    """
    model.eval()
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"Failed to load wikitext dataset: {e}")
        return float('inf')

    text_list = [text for text in dataset['text'] if text.strip()][:num_texts]
    if not text_list:
        print("Warning: No valid texts found in the dataset sample.")
        return float('inf')

    full_text = "\n\n".join(text_list)
    encodings = tokenizer(full_text, return_tensors="pt")

    seq_len = encodings.input_ids.size(1)
    nlls = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        if end_loc - begin_loc < 1:
            continue
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
    if not nlls:
        return float('inf')
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

def run_layer_replacement_analysis():
    """
    Main function to perform layer replacement analysis. For each layer 'i', it is
    replaced by a copy of layer 'j', and the resulting model's perplexity is measured.
    """
    # --- 1. CONFIGURATION ---
    MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE} with dtype: {DTYPE}")
    print(f"Loading model: {MODEL_NAME}. This may take a moment...")

    # --- 2. LOAD ORIGINAL MODEL AND TOKENIZER ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # KEY CHANGE: Load the model to CPU first to act as our 'source'
        original_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="cpu", # Load to system RAM
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    num_layers = original_model.config.num_hidden_layers
    print(f"Model loaded to CPU. It has {num_layers} layers.")

    # Define layers to exclude from the analysis (e.g., first few and last)
    layers_to_skip = {0, 1, 2, num_layers - 1}
    print(f"Skipping analysis for layers: {sorted(list(layers_to_skip))}")

    # --- 3. GET BASELINE PERPLEXITY ---
    results_matrix = np.full((num_layers, num_layers), np.nan)
    print("\nCalculating baseline perplexity (original model)...")
    
    # Temporarily move model to GPU for evaluation
    original_model.to(DEVICE)
    baseline_ppl = evaluate_perplexity(original_model, tokenizer, DEVICE, num_texts=10)
    
    # KEY CHANGE: Move model back to CPU to free VRAM for the analysis loop
    original_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Baseline Perplexity: {baseline_ppl:.4f}")
    np.fill_diagonal(results_matrix, baseline_ppl)

    # --- 4. ITERATE THROUGH ALL (TARGET, SOURCE) PAIRS ---
    print("\nStarting layer replacement analysis. This will be slow due to CPU->GPU transfers.")
    
    num_active_layers = num_layers - len(layers_to_skip)
    total_iterations = num_active_layers * (num_active_layers - 1) # Total non-diagonal replacements
    pbar = tqdm(total=total_iterations, desc="Experiments", ncols=100)

    for target_layer_idx in range(num_layers):
        if target_layer_idx in layers_to_skip:
            continue

        for source_layer_idx in range(num_layers):
            if source_layer_idx in layers_to_skip:
                continue
            
            if target_layer_idx == source_layer_idx:
                continue

            pbar.set_description(f"Replacing {target_layer_idx}<-{source_layer_idx}")

            # Create a deep copy of the model ON THE CPU
            model_copy = copy.deepcopy(original_model)
            
            # Replace the target layer with a copy of the source layer (still on CPU)
            model_copy.model.layers[target_layer_idx] = copy.deepcopy(
                original_model.model.layers[source_layer_idx]
            )
            
            # KEY CHANGE: Move the modified copy to the GPU for evaluation
            model_copy.to(DEVICE)
            
            ppl = evaluate_perplexity(model_copy, tokenizer, DEVICE, num_texts=3)
            print(ppl)
            ppl = min(50, ppl)
            print(ppl)
            results_matrix[target_layer_idx, source_layer_idx] = ppl
            
            # Clean up memory, especially the large model_copy on the GPU
            del model_copy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pbar.update(1)
    pbar.close()
    print(results_matrix)
    # --- 5. ANALYZE AND VISUALIZE THE RESULTS ---
    print("\nAnalysis complete. Generating results...")
    best_replacements_indices = np.nanargmin(results_matrix, axis=1)
    
    print("\n--- Best Replacement for Each Layer (excluding skipped layers) ---")
    for i in range(num_layers):
        if i in layers_to_skip:
            continue
            
        best_j = best_replacements_indices[i]
        original_ppl = results_matrix[i, i]
        best_ppl = results_matrix[i, best_j]
        
        if not np.isnan(original_ppl) and not np.isnan(best_ppl):
            print(f"Layer {i:2d}: Best replaced by layer {best_j:2d}. "
                  f"PPL changes from {original_ppl:.2f} to {best_ppl:.2f}.")
        else:
             print(f"Layer {i:2d}: Data not available (likely skipped).")

    # Generate and save the heatmap
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 15))
    cmap = "viridis_r"
    
    sns.heatmap(results_matrix, 
                ax=ax, 
                cmap=cmap,
                annot=False,
                xticklabels=range(num_layers),
                yticklabels=range(num_layers),
                vmin=np.nanmin(results_matrix), # Use actual min perplexity for better color scale
                vmax=np.nanpercentile(results_matrix, 98)) # Cap at 98th percentile to avoid outliers ruining scale

    ax.set_title(f'Perplexity after Replacing Layer i with Layer j in {MODEL_NAME}', fontsize=20, pad=20)
    ax.set_ylabel('Target Layer Index (Replaced At)', fontsize=14)
    ax.set_xlabel('Source Layer Index (Copied From)', fontsize=14)

    for layer_idx in layers_to_skip:
        ax.axhline(y=layer_idx + 0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(x=layer_idx + 0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.plot([0, num_layers], [0, num_layers], color='red', linestyle=':', linewidth=1.5, alpha=0.7)

    cbar = ax.collections[0].colorbar
    cbar.set_label('Perplexity (Lower is Better)', rotation=270, labelpad=25, fontsize=14)
    fig.tight_layout()

    # Save the figure to a file
    filename = f"perplexity_replacement_heatmap_{MODEL_NAME.replace('/', '-')}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nHeatmap saved as '{filename}'")
    


if __name__ == '__main__':
    run_layer_replacement_analysis()
