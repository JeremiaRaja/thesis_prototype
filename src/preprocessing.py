"""
preprocessing.py
────────────────
Clean tweets from CSV splits produced by split_data.py.
Applies cleaning + normalization and overwrites the CSVs with clean_text column.

Usage:
    python -m src.preprocessing
"""

import re
import unicodedata
import pandas as pd
import os
from src.normalization import normalize
from src.config import FORMAL_PROCESSED, INFORMAL_PROCESSED


def remove_url(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_mention(text):
    return re.sub(r"@\w+", "", text)

def remove_hashtag(text):
    return re.sub(r"#(\w+)", r"\1", text)

def remove_emoji(text):
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sm") and ord(ch) <= 0xFFFF
    )

def remove_punctuation(text):
    return re.sub(r"[^\w\s]", " ", text)

def remove_numbers(text):
    return re.sub(r"\d+", "", text)

def clean_tweet(text):
    text = str(text)
    text = remove_url(text)
    text = remove_mention(text)
    text = remove_hashtag(text)
    text = remove_emoji(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize(text)
    return text.strip()


def clean_split_csvs(processed_dir: str, name: str):
    """Clean all split CSVs in a processed directory."""
    for split in ['train', 'val', 'test']:
        path = os.path.join(processed_dir, f"{split}.csv")
        df = pd.read_csv(path)
        df['clean_text'] = df['text'].apply(clean_tweet)
        df = df[df['clean_text'].str.strip() != ""].reset_index(drop=True)
        df.to_csv(path, index=False)
        print(f"[preprocessing] {name:8s} {split:5s}: {len(df):>5,} samples cleaned")


if __name__ == "__main__":
    print("[preprocessing] Cleaning FORMAL splits...")
    clean_split_csvs(FORMAL_PROCESSED, "FORMAL")

    print("[preprocessing] Cleaning INFORMAL splits...")
    clean_split_csvs(INFORMAL_PROCESSED, "INFORMAL")

    print("[preprocessing] Done!")
