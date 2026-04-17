"""
data_loader.py
──────────────
PyTorch Dataset wrapper and DataLoader factory for BERT fine-tuning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from src.config import PRETRAINED_MODEL, MAX_LEN, BATCH_SIZE


class TweetDataset(Dataset):
    """Map-style dataset for tokenized tweets."""

    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.texts     = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels    = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_tokenizer(model_name: str = PRETRAINED_MODEL):
    return AutoTokenizer.from_pretrained(model_name)


def make_dataloaders(splits: dict, tokenizer, batch_size: int = BATCH_SIZE):
    """
    Args:
        splits: dict returned by preprocessing.split_dataset()
                {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        tokenizer: HuggingFace tokenizer
        batch_size: samples per batch

    Returns:
        dict of {split_name: DataLoader}
    """
    loaders = {}
    for name, (X, y) in splits.items():
        ds = TweetDataset(X, y, tokenizer)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,          # safe for Windows / Colab
            pin_memory=torch.cuda.is_available(),
        )
        print(f"[data_loader] {name:5s}: {len(ds):>5,} samples | "
              f"{len(loaders[name]):>4} batches")
    return loaders


# ─── Load from pre-saved CSVs (alternative entry point) ──────────────────────

def load_from_csv(data_dir: str, tokenizer, batch_size: int = BATCH_SIZE):
    """Load train/val/test CSVs saved by preprocessing.split_dataset()."""
    import os
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{name}.csv")
        df = pd.read_csv(path)
        splits[name] = (df["clean_text"], df["label"])
    return make_dataloaders(splits, tokenizer, batch_size)
