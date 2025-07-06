import torch
import warnings
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Optional, Union

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore")

def evaluate_model_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: Optional[str] = None,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "test",
    num_samples: Optional[int] = 50,
    max_length: int = 1024,
    stride: int = 512
) -> float:
    """
    Calculates the perplexity of a Hugging Face model on a specified dataset.

    This function implements a standard sliding window evaluation to handle long texts
    without exceeding the model's context limit. It's a robust way to measure
    how well a language model "understands" a body of text.

    Args:
        model (PreTrainedModel): The Hugging Face model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        device (Optional[str]): The device to run the evaluation on (e.g., "cuda", "cpu").
            If None, it will auto-detect CUDA availability.
        dataset_name (str): The name of the dataset to use from the Hugging Face Hub.
            Defaults to "wikitext".
        dataset_config (str): The specific configuration of the dataset.
            Defaults to "wikitext-103-raw-v1", a standard benchmark.
        split (str): The dataset split to use (e.g., "test", "validation").
            Defaults to "test".
        num_samples (Optional[int]): The number of text samples to use from the dataset.
            If None, the entire dataset split will be used (can be very slow).
            Defaults to 50 for a quick, representative score.
        max_length (int): The maximum context length the model can handle in a single pass.
            Defaults to 1024.
        stride (int): The number of tokens to slide the context window by.
            A stride smaller than max_length ensures overlapping context.
            Defaults to 512.

    Returns:
        float: The calculated perplexity score. Lower is better. Returns float('inf')
               if no valid text is found in the dataset subset.
    """
    # 1. Setup and Sanity Checks
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Perplexity Evaluation on device: {device} ---")
    
    model.eval()
    model.to(device)

    # 2. Load and Prepare the Dataset
    print(f"Loading dataset: '{dataset_name}' (config: '{dataset_config}', split: '{split}')...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # Select a subset of the data if num_samples is specified
    if num_samples is not None:
        print(f"Using {num_samples} samples from the dataset.")
        dataset = dataset.select(range(num_samples))
        
    # Filter out empty texts and concatenate into a single string
    text_list = [text for text in dataset['text'] if text.strip()]
    if not text_list:
        print("Warning: No valid text found in the selected dataset samples.")
        return float('inf')
        
    full_text = "\n\n".join(text_list)
    
    # 3. Tokenize the entire text
    print("Tokenizing the text...")
    encodings = tokenizer(full_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    # 4. Calculate Perplexity using a Sliding Window
    nlls = [] # To store negative log likelihoods for each window
    
    print(f"Calculating perplexity with max_length={max_length} and stride={stride}...")
    pbar = tqdm(range(0, seq_len, stride), desc="Calculating Perplexity", ncols=100)
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        
        # Ensure that the last segment is not too small to be meaningful
        if end_loc - begin_loc < stride and begin_loc > 0:
            continue
            
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        # The model calculates loss by shifting input_ids internally
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            # outputs.loss is the average negative log likelihood
            nlls.append(outputs.loss)
    
    # 5. Compute and Return Final Perplexity
    if not nlls:
        print("Warning: Could not calculate perplexity. The text might be shorter than the stride.")
        return float('inf')
        
    perplexity = torch.exp(torch.stack(nlls).mean()).item()
    
    print("-" * 50)
    print(f"✅ Final Perplexity: {perplexity:.4f}")
    print("-" * 50)
    
    return perplexity

# --- Example Usage ---
if __name__ == '__main__':
    # You can stick any Hugging Face Causal LM and its tokenizer here.
    MODEL_NAME = "gpt2" # Using gpt2 for a quick and small example
    
    print(f"Loading model '{MODEL_NAME}' for demonstration...")
    # For larger models, you might want to add: torch_dtype="auto"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add a padding token if it doesn't exist (like in GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Now, anyone on the team can just call the function like this:
    ppl_score = evaluate_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        num_samples=10, # Using a small number of samples for a quick demo
        max_length=512, # GPT-2 has a smaller context
        stride=256
    )
    
    print(f"\nThe function returned a perplexity score of: {ppl_score:.4f}")
