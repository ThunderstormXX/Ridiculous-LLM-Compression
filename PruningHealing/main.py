from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_healing.iterative_runner import run_iterative_pruning_and_training
from lora_healing.train import train
from lora_healing.eval import evaluate_model_perplexity

if __name__ == "__main__":
    model_path = "/home/user31/polina/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cuda:3")

    run_iterative_pruning_and_training(
        base_model=model,
        tokenizer=tokenizer,
        start_layer = 15, 
        num_layers_to_prune= 5, 
        train_fn=train,
        eval_fn=evaluate_model_perplexity
    )
