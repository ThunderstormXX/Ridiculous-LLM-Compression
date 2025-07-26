import torch
import warnings
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from typing import Optional

warnings.filterwarnings("ignore")

from datasets import load_dataset, DatasetDict, Dataset
import os
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional

CACHE_PATH = "./cached_eval_data"

def get_local_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_samples: Optional[int]
) -> Dataset:
    os.makedirs(CACHE_PATH, exist_ok=True)
    filename = f"{dataset_name.replace('/', '_')}_{dataset_config}_{split}_{num_samples}.arrow"
    cache_file = os.path.join(CACHE_PATH, filename)

    if os.path.exists(cache_file):
        print(f"✅ Loading cached dataset from: {cache_file}")
        return Dataset.load_from_disk(cache_file)

    print("⬇️ Downloading dataset and caching subset...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    dataset.save_to_disk(cache_file)
    return dataset


def evaluate_model_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: Optional[str] = None,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "test",
    num_samples: Optional[int] = 1000,
    max_length: int = 1024,
    stride: int = 512
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Perplexity Evaluation on device: {device} ---")

    model.eval()
    if next(model.parameters()).device.type != device:
        model.to(device)

    print(f"Loading dataset: '{dataset_name}' (config: '{dataset_config}', split: '{split}')...")
    dataset = get_local_dataset(dataset_name, dataset_config, split, num_samples)

    text_list = [text for text in dataset["text"] if text.strip()]
    if not text_list:
        print("Warning: No valid text found.")
        return float("inf")

    full_text = "\n\n".join(text_list)
    print("Tokenizing the text...")
    encodings = tokenizer(full_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    print(f"Calculating perplexity with max_length={max_length} and stride={stride}...")
    pbar = tqdm(range(0, seq_len, stride), desc="Calculating Perplexity", ncols=100)

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        if end_loc - begin_loc < stride and begin_loc > 0:
            continue

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

    if not nlls:
        print("Warning: Could not calculate perplexity.")
        return float("inf")

    perplexity = torch.exp(torch.stack(nlls).mean()).item()

    print("-" * 50)
    print(f"✅ Final Perplexity: {perplexity:.4f}")
    print("-" * 50)

    torch.cuda.empty_cache()
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