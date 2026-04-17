"""
compare.py
──────────
Compare bert-base-multilingual-cased (mBERT) against classical baselines:
  • Naïve Bayes (TF-IDF + Complement NB)
  • SVM         (TF-IDF + LinearSVC)
  • Logistic Regression (TF-IDF)

Produces a comparison table (CSV) + bar chart saved to results/.

Usage:
    python -m src.compare
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

from src.config import DATA_PROCESSED, RESULTS_DIR, LABEL_MAP, MODEL_SHORT_NAME


# ─── Load preprocessed CSV splits ────────────────────────────────────────────

def load_splits():
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(DATA_PROCESSED, f"{name}.csv")
        df = pd.read_csv(path)
        splits[name] = (df["clean_text"], df["label"])
    return splits


# ─── Baseline models ─────────────────────────────────────────────────────────

def make_baselines():
    tfidf = dict(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   ComplementNB()),
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   LinearSVC(max_iter=2000, C=1.0)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
        ]),
    }


# ─── Parse accuracy + macro-F1 from a classification_report text file ────────

def _parse_report(path: str):
    acc, f1 = None, None
    with open(path) as f:
        for line in f:
            if "accuracy" in line.lower():
                parts = line.split()
                try:
                    acc = float(parts[-2])
                except (ValueError, IndexError):
                    pass
            if "macro avg" in line.lower():
                parts = line.split()
                try:
                    f1 = float(parts[-2])
                except (ValueError, IndexError):
                    pass
    return acc, f1


# ─── Run comparison ───────────────────────────────────────────────────────────

def compare():
    splits = load_splits()
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # Combine train + val for baseline fitting (no val set needed)
    X_fit = pd.concat([X_train, X_val], ignore_index=True)
    y_fit = pd.concat([y_train, y_val], ignore_index=True)

    results     = []
    label_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

    # ── Baselines ─────────────────────────────────────────────────────────────
    for name, pipe in make_baselines().items():
        print(f"[compare] Training {name} …")
        pipe.fit(X_fit, y_fit)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="macro")
        f1_w = f1_score(y_test, y_pred, average="weighted")

        results.append({"Model": name, "Accuracy": acc,
                        "Macro-F1": f1, "Weighted-F1": f1_w})
        print(f"  acc={acc:.4f}  macro-f1={f1:.4f}")

        slug = name.replace(" ", "_").lower()
        report_txt = classification_report(y_test, y_pred,
                                           target_names=label_names)
        with open(os.path.join(RESULTS_DIR, f"report_{slug}.txt"), "w") as f:
            f.write(report_txt)

    # ── mBERT results (if already trained) ───────────────────────────────────
    bert_report = os.path.join(RESULTS_DIR,
                               f"classification_report_{MODEL_SHORT_NAME}.txt")
    if os.path.exists(bert_report):
        bert_acc, bert_f1 = _parse_report(bert_report)
        results.append({
            "Model":        f"{MODEL_SHORT_NAME} (fine-tuned)",
            "Accuracy":     bert_acc,
            "Macro-F1":     bert_f1,
            "Weighted-F1":  None,
        })
        print(f"[compare] {MODEL_SHORT_NAME} → "
              f"acc={bert_acc:.4f}  macro-f1={bert_f1:.4f}")
    else:
        print(f"[compare] {MODEL_SHORT_NAME} results not found "
              f"— run `python -m src.train` first.")

    # ── Comparison table ──────────────────────────────────────────────────────
    df_results = pd.DataFrame(results).set_index("Model")
    print("\n" + "─" * 60)
    print(f"Model Comparison — {MODEL_SHORT_NAME} vs Baselines (Test Set)")
    print("─" * 60)
    print(df_results.to_string(float_format="{:.4f}".format))
    print("─" * 60)

    df_results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    df_plot = df_results[["Accuracy", "Macro-F1"]].dropna()
    x       = np.arange(len(df_plot))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, df_plot["Accuracy"], width,
                   label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, df_plot["Macro-F1"], width,
                   label="Macro-F1", color="#DD8452")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Sentiment Analysis — {MODEL_SHORT_NAME} vs Baselines (PPKM Tweets)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[compare] Bar chart saved → {chart_path}")

    return df_results


if __name__ == "__main__":
    compare()