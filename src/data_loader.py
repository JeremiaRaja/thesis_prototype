"""
data_loader.py — PyTorch Dataset & DataLoader for IndoBERT
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import os
from src.config import PRETRAINED_MODEL, MAX_LEN, BATCH_SIZE


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_tokenizer():
    return AutoTokenizer.from_pretrained(PRETRAINED_MODEL)


def load_from_csv(processed_dir: str, tokenizer, batch_size=BATCH_SIZE):
    """Load train/val/test CSVs and return DataLoaders."""
    loaders = {}
    for name in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(processed_dir, f"{name}.csv"))
        ds = TweetDataset(df['clean_text'], df['label'], tokenizer)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == 'train'),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"[data_loader] {name:5s}: {len(ds):>5,} samples | "
              f"{len(loaders[name]):>4} batches")
    return loaders
