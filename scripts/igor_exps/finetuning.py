# scripts/finetuning.py
import argparse
from src.pruninghealing.utils import load_model_and_tokenizer
from src.pruninghealing.trainer import Trainer
from src.pruninghealing.dataset import DatasetLoader

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on dataset")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name or path")
    parser.add_argument("--output_path", required=True, help="Path to save fine-tuned model")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Load dataset
    dataset_loader = DatasetLoader(tokenizer)
    if args.dataset == "wikitext":
        dataset_loader.load_wikitext()
    else:
        dataset_loader.load_custom(args.dataset)
    
    # Create trainer and train
    trainer = Trainer(model, tokenizer, args.workspace)
    trained_model = trainer.train(
        dataset_loader, 
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # Save fine-tuned model
    trained_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print(f"Fine-tuned model saved to: {args.output_path}")

if __name__ == "__main__":
    main()