from datasets import load_dataset, Dataset, DatasetDict
import os

def load_and_cache_dataset(dataset_name, splits_with_sizes, save_dir):
    """
    Загружает из streaming указанные split'ы с ограничением по числу примеров,
    собирает их в DatasetDict и сохраняет в одну папку.
    
    splits_with_sizes: dict вида {split_name: num_samples}
    """
    datasets = {}
    for split, size in splits_with_sizes.items():
        print(f"Loading {size} samples from {dataset_name} split '{split}' (streaming)...")
        stream = load_dataset(dataset_name, "en", split=split, streaming=True)

        samples = []
        for i, sample in enumerate(stream):
            if i >= size:
                break
            samples.append(sample)

        ds = Dataset.from_list(samples)
        datasets[split] = ds
        print(f"Loaded {len(ds)} samples for split '{split}'.")

    dataset_dict = DatasetDict(datasets)

    os.makedirs(save_dir, exist_ok=True)
    abs_path = os.path.abspath(save_dir)
    print(f"Saving DatasetDict to {abs_path} ...")
    dataset_dict.save_to_disk(abs_path)
    print("Saved successfully.")

if __name__ == "__main__":
    save_path = "./cached_dataset"
    splits_and_sizes = {
        "train": 50000,
        "validation": 1000,
    }
    load_and_cache_dataset("allenai/c4", splits_and_sizes, save_path)
