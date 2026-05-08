"""
split_data.py
─────────────
Split dataset.xlsx into:
  - dataset_formal.xlsx   (tweets using standard Indonesian)
  - dataset_informal.xlsx (tweets using slang / informal Indonesian)

Then further split each into train / val / test CSV files.

Usage:
    python -m src.split_data
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_FILE, FORMAL_FILE, INFORMAL_FILE,
    FORMAL_PROCESSED, INFORMAL_PROCESSED,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE
)

# ─── Slang word list for informal detection ───────────────────────────────────
SLANG_WORDS = set([
    'gak','ga','ngga','nggak','gue','gw','lo','lu','udah','udh',
    'kalo','aja','sih','deh','dong','lah','bgt','banget','emg',
    'emang','yg','dgn','tdk','sdh','blm','msh','krn','gmn',
    'gimana','kayak','kyk','tp','sampe','ampe','wkwk','haha',
    'hehe','nih','guys','gabisa','gausa','gasuka','gausah',
    'gasalah','gaada','gitu','gini','gt','sy','km','mrk','kt',
    'pengen','pengin','kesel','capek','males','malas','nanya',
    'bikin','liat','bakal','kok','kan','tuh','plis','please',
])


def is_informal(text: str, min_slang: int = 2) -> bool:
    """Return True if tweet contains >= min_slang slang words."""
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    return sum(1 for t in tokens if t in SLANG_WORDS) >= min_slang


def split_formal_informal(filepath: str = RAW_DATA_FILE):
    """Split raw dataset into formal and informal Excel files."""
    print(f"[split_data] Reading: {filepath}")
    df = pd.read_excel(filepath)
    df = df[['Date', 'User', 'Tweet', 'sentiment']].dropna()
    df['sentiment'] = df['sentiment'].astype(int)

    df['is_informal'] = df['Tweet'].apply(is_informal)
    formal_df   = df[~df['is_informal']].drop(columns='is_informal').reset_index(drop=True)
    informal_df = df[ df['is_informal']].drop(columns='is_informal').reset_index(drop=True)

    formal_df.to_excel(FORMAL_FILE,   index=False)
    informal_df.to_excel(INFORMAL_FILE, index=False)

    print(f"[split_data] Formal   : {len(formal_df):,} tweets → {FORMAL_FILE}")
    print(f"[split_data] Informal : {len(informal_df):,} tweets → {INFORMAL_FILE}")
    print(f"\n[split_data] Formal label distribution:")
    print(formal_df['sentiment'].value_counts().sort_index())
    print(f"\n[split_data] Informal label distribution:")
    print(informal_df['sentiment'].value_counts().sort_index())

    return formal_df, informal_df


def make_splits(df: pd.DataFrame, out_dir: str, name: str):
    """Split dataframe into train/val/test CSVs saved to out_dir."""
    X, y = df['Tweet'], df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_train, random_state=RANDOM_STATE)

    splits = {
        'train': (X_train, y_train),
        'val':   (X_val,   y_val),
        'test':  (X_test,  y_test),
    }
    for split_name, (X_, y_) in splits.items():
        out = pd.DataFrame({'text': X_.reset_index(drop=True),
                            'label': y_.reset_index(drop=True)})
        path = os.path.join(out_dir, f"{split_name}.csv")
        out.to_csv(path, index=False)
        print(f"[split_data] {name:8s} {split_name:5s}: {len(out):>5,} samples → {path}")


if __name__ == "__main__":
    formal_df, informal_df = split_formal_informal()

    print("\n[split_data] Creating train/val/test splits...")
    make_splits(formal_df,   FORMAL_PROCESSED,   "FORMAL")
    make_splits(informal_df, INFORMAL_PROCESSED, "INFORMAL")
    print("\n[split_data] Done!")
