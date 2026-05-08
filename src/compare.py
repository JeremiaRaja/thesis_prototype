"""
compare.py
──────────
Compare IndoBERT (formal) vs IndoBERT (informal) vs baselines.
Reads saved classification reports and produces a combined bar chart.

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
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import (
    FORMAL_PROCESSED, INFORMAL_PROCESSED,
    RESULTS_DIR, LABEL_MAP, MODEL_SHORT_NAME
)


def load_csv_splits(processed_dir):
    splits = {}
    for name in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(processed_dir, f"{name}.csv"))
        splits[name] = (df['clean_text'], df['label'])
    return splits


def make_baselines():
    tfidf = dict(ngram_range=(1,2), max_features=50_000, sublinear_tf=True)
    return {
        "Naive Bayes":        Pipeline([("tfidf", TfidfVectorizer(**tfidf)),
                                        ("clf",   ComplementNB())]),
        "SVM":                Pipeline([("tfidf", TfidfVectorizer(**tfidf)),
                                        ("clf",   LinearSVC(max_iter=2000))]),
        "Logistic Regression":Pipeline([("tfidf", TfidfVectorizer(**tfidf)),
                                        ("clf",   LogisticRegression(max_iter=1000,
                                                                     solver="lbfgs"))]),
    }


def run_baselines(processed_dir, dataset_name):
    splits   = load_csv_splits(processed_dir)
    X_fit    = pd.concat([splits['train'][0], splits['val'][0]], ignore_index=True)
    y_fit    = pd.concat([splits['train'][1], splits['val'][1]], ignore_index=True)
    X_test, y_test = splits['test']
    label_names    = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    results  = []

    for name, pipe in make_baselines().items():
        pipe.fit(X_fit, y_fit)
        y_pred = pipe.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro")
        results.append({"Model": f"{name} ({dataset_name})",
                        "Accuracy": acc, "Macro-F1": f1})
        report = classification_report(y_test, y_pred, target_names=label_names)
        slug   = name.replace(" ","_").lower()
        with open(os.path.join(RESULTS_DIR,
                               f"report_{slug}_{dataset_name}.txt"), "w") as f:
            f.write(report)
        print(f"  {name:22s} ({dataset_name}): acc={acc:.4f}  f1={f1:.4f}")
    return results


def parse_report(path):
    acc = f1 = None
    with open(path) as f:
        for line in f:
            if "accuracy" in line.lower():
                try: acc = float(line.split()[-2])
                except: pass
            if "macro avg" in line.lower():
                try: f1 = float(line.split()[-2])
                except: pass
    return acc, f1


def compare():
    results = []

    print("\n[compare] Running baselines on FORMAL data...")
    results += run_baselines(FORMAL_PROCESSED, "formal")

    print("\n[compare] Running baselines on INFORMAL data...")
    results += run_baselines(INFORMAL_PROCESSED, "informal")

    # Load IndoBERT results
    for mode in ['formal', 'informal']:
        tag  = f"{MODEL_SHORT_NAME}_{mode}"
        path = os.path.join(RESULTS_DIR, f"report_{tag}.txt")
        if os.path.exists(path):
            acc, f1 = parse_report(path)
            results.append({"Model": f"{MODEL_SHORT_NAME} ({mode})",
                            "Accuracy": acc, "Macro-F1": f1})
            print(f"\n[compare] {MODEL_SHORT_NAME} ({mode}): "
                  f"acc={acc:.4f}  f1={f1:.4f}")
        else:
            print(f"\n[compare] {tag} not found — run `python -m src.train --mode {mode}` first.")

    # Table
    df_results = pd.DataFrame(results).set_index("Model")
    print("\n" + "─"*65)
    print(f"Full Model Comparison")
    print("─"*65)
    print(df_results.to_string(float_format="{:.4f}".format))
    print("─"*65)
    df_results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))

    # Bar chart
    df_plot = df_results[["Accuracy","Macro-F1"]].dropna()
    x       = np.arange(len(df_plot))
    width   = 0.35
    colors  = {"formal": "#4C72B0", "informal": "#DD8452"}

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1   = ax.bar(x - width/2, df_plot["Accuracy"], width,
                     label="Accuracy", color="#4C72B0", alpha=0.85)
    bars2   = ax.bar(x + width/2, df_plot["Macro-F1"], width,
                     label="Macro-F1", color="#DD8452", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"{MODEL_SHORT_NAME} Formal vs Informal — Full Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[compare] Chart saved → {chart_path}")

    return df_results


if __name__ == "__main__":
    compare()
