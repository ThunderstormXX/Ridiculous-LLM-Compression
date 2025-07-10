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

def run_double_removal_analysis():
    """
    Main function to perform the double-layer removal analysis on Llama-3.2-3B-Instruct,
    calculate perplexity for each modification, and generate a heatmap of the results.
    """
    # --- 1. CONFIGURATION ---
    MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE} with dtype: {DTYPE}")
    print(f"Loading model: {MODEL_NAME}. This may take a while...")

    # --- 2. LOAD ORIGINAL MODEL AND TOKENIZER ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # --- KEY CHANGE 1: Load the model to CPU first to act as our 'source' ---
        # This prevents OOM errors from trying to deepcopy a model on the GPU.
        original_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="cpu", # Load to system RAM
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    num_layers = original_model.config.num_hidden_layers
    print(f"Model loaded successfully to CPU. It has {num_layers} layers.")

    layers_to_skip = {0, 1, 2, num_layers - 1}
    print(f"Skipping analysis for layers: {sorted(list(layers_to_skip))}")
    # --- 3. CALCULATE BASELINE PERPLEXITY ---
    print("\nCalculating baseline perplexity for the original model...")
    # Temporarily move the model to GPU for evaluation
    original_model.to(DEVICE)
    baseline_ppl = evaluate_perplexity(original_model, tokenizer, DEVICE, num_texts=10)
    # --- KEY CHANGE 2: Move the model back to CPU to free up VRAM ---
    original_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if baseline_ppl == float('inf'):
        print("Could not calculate baseline perplexity. Aborting.")
        return
    print(f"Baseline Perplexity (original model): {baseline_ppl:.4f}")

    # --- 4. INITIALIZE RESULTS MATRIX ---
    results_matrix = np.full((num_layers, num_layers), np.nan)

    # --- 5. ITERATE THROUGH ALL UNIQUE PAIRS OF LAYERS ---
    print("\nStarting double-layer removal analysis. This will be slow due to CPU->GPU transfers.")
    total_iterations = num_layers * (num_layers - 1) // 2
    pbar = tqdm(total=total_iterations, desc="Analyzing layer pairs", ncols=100)

    for i in range(num_layers):
        if i in layers_to_skip:
            continue
        for j in range(i + 1, num_layers):
            if j in layers_to_skip:
                continue
            
            pbar.set_description(f"Removing Layers ({i}, {j})")

            # Create a deep copy of the model ON THE CPU
            model_copy = copy.deepcopy(original_model)
            
            # --- KEY CHANGE 3: Move the new copy to the GPU for this iteration ---
            model_copy.to(DEVICE)
            
            layers = model_copy.model.layers
            del layers[j]
            del layers[i]
            model_copy.config.num_hidden_layers = len(layers)
            
            ppl = evaluate_perplexity(model_copy, tokenizer, DEVICE, num_texts=3)
            results_matrix[i, j] = ppl
            results_matrix[j, i] = ppl

            # Clean up memory, especially the large model_copy on the GPU
            del model_copy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pbar.update(1)
    
    pbar.close()

    # --- 6. VISUALIZE THE RESULTS ---
    print("\nAnalysis complete. Generating and saving heatmap...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 15))

    np.fill_diagonal(results_matrix, baseline_ppl)
    mask = np.triu(np.ones_like(results_matrix, dtype=bool))
    
    sns.heatmap(results_matrix,
                ax=ax,
                cmap="viridis_r",
                annot=False,
                mask=mask,
                xticklabels=range(num_layers),
                yticklabels=range(num_layers))

    ax.set_title(f'Perplexity After Removing Two Layers from {MODEL_NAME}', fontsize=20, pad=20)
    ax.set_xlabel('Index of Second Removed Layer', fontsize=14)
    ax.set_ylabel('Index of First Removed Layer', fontsize=14)
    
    cbar = ax.collections[0].colorbar
    cbar.set_label('Perplexity (Higher is Worse)', rotation=270, labelpad=25, fontsize=14)
    fig.tight_layout()

    filename = f"perplexity_heatmap_{MODEL_NAME.replace('/', '-')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Heatmap saved as '{filename}'")

if __name__ == '__main__':
    run_double_removal_analysis()
