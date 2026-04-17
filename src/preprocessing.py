"""
preprocessing.py
────────────────
Tweet cleaning pipeline for Indonesian PPKM sentiment data.
Removes noise (URLs, mentions, hashtags, emoji, etc.) before tokenization.
"""

import re
import unicodedata
import pandas as pd
from src.normalization import normalize
from src.config import (
    RAW_DATA_FILE, DATA_PROCESSED, LABEL_MAP,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE
)
import os
from sklearn.model_selection import train_test_split


# ─── Cleaning steps ───────────────────────────────────────────────────────────

def remove_url(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_mention(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def remove_hashtag(text: str) -> str:
    """Keep the word, drop the '#' symbol."""
    return re.sub(r"#(\w+)", r"\1", text)


def remove_emoji(text: str) -> str:
    """Remove emoji and other non-BMP unicode."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sm") and ord(ch) <= 0xFFFF
    )


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)


def clean_tweet(text: str) -> str:
    """
    Full cleaning pipeline (applied before BERT tokenization):
    URL → mention → hashtag → emoji → punctuation → numbers → normalize → trim
    """
    text = str(text)
    text = remove_url(text)
    text = remove_mention(text)
    text = remove_hashtag(text)
    text = remove_emoji(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize(text)          # lowercase + slang + repeated chars
    return text.strip()


# ─── Dataset loading & splitting ─────────────────────────────────────────────

def load_and_clean(filepath: str = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw xlsx, clean tweets, drop nulls/duplicates."""
    df = pd.read_excel(filepath)
    df = df[["Tweet", "sentiment"]].copy()
    df.columns = ["text", "label"]

    # Drop missing / duplicate rows
    df.dropna(subset=["text", "label"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)

    # Ensure labels are integers 0/1/2
    df["label"] = df["label"].astype(int)
    assert df["label"].isin([0, 1, 2]).all(), "Unexpected label values found!"

    df["clean_text"] = df["text"].apply(clean_tweet)

    # Drop rows where cleaning produced empty string
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    print(f"[preprocessing] Total samples after cleaning : {len(df):,}")
    print(f"[preprocessing] Label distribution:\n{df['label'].value_counts().sort_index()}")
    return df


def split_dataset(df: pd.DataFrame):
    """
    Stratified split → train / validation / test.
    Default: 70 % train | 15 % val | 15 % test
    """
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_train, random_state=RANDOM_STATE
    )

    splits = {
        "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        "val":   (X_val.reset_index(drop=True),   y_val.reset_index(drop=True)),
        "test":  (X_test.reset_index(drop=True),  y_test.reset_index(drop=True)),
    }

    for name, (X_, y_) in splits.items():
        print(f"[preprocessing] {name:5s}: {len(X_):>5,} samples")
        out = pd.DataFrame({"clean_text": X_, "label": y_})
        out.to_csv(os.path.join(DATA_PROCESSED, f"{name}.csv"), index=False)

    return splits


if __name__ == "__main__":
    df = load_and_clean()
    split_dataset(df)
    print("[preprocessing] Done — CSVs saved to data/processed/")
