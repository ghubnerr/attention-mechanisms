from pathlib import Path
import argparse

import numpy as np
import jax
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


def batch_dataset(dataset, batch_size):
    batch = {"input_ids": []}
    for example in dataset:
        batch["input_ids"].append(example["input_ids"])
        if len(batch["input_ids"]) == batch_size:
            yield {k: np.array(v, dtype=np.int32) for k, v in batch.items()}
            batch = {"input_ids": []}

    if batch["input_ids"]:
        yield {k: np.array(v, dtype=np.int32) for k, v in batch.items()}


def shard_batch(batch):
    batch_size = batch["input_ids"].shape[0]
    per_device_batch = batch_size // jax.device_count()
    return jax.tree.map(lambda x: x.reshape((jax.device_count(), per_device_batch, -1)), batch)


def load_dataset(dataset_path):
    print(f"Attempting to load dataset from: {dataset_path}")
    if dataset_path and Path(dataset_path).exists():
        try:
            return load_from_disk(dataset_path)
        except Exception as e:
            print(
                f"Could not load dataset from disk using Hugging Face 'load_from_disk': {e}")
            print(
                "Please ensure your dataset is in the correct format or implement custom loading.")
            raise


def build_openweb(split: str = "train", seq_len: int = 1024) -> DatasetDict:
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    ds = load_dataset("openwebtext", split=split)

    def tokenize_fn(examples):
        tokens = tok(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )
        return {"input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]}

    ds = ds.map(tokenize_fn,
                batched=True,
                remove_columns=["text"],
                num_proc=4)
    ds = ds.shuffle(seed=42)
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="Directory in which to save the processed dataset")
    args = parser.parse_args()

    out_dir = Path(args.path).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_openweb()
    # Arrow-based, reload with load_from_disk
    dataset.save_to_disk(out_dir)
    print(f"Saved OpenWebText to {out_dir}")
