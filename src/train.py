"""
train.py
────────
Fine-tune IndoBERT on either FORMAL or INFORMAL tweet dataset.

Usage:
    python -m src.train --mode formal
    python -m src.train --mode informal
"""

import os
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PRETRAINED_MODEL, MODEL_SHORT_NAME, NUM_LABELS, LABEL_MAP,
    EPOCHS, LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY, DROPOUT,
    BATCH_SIZE, FORMAL_PROCESSED, INFORMAL_PROCESSED,
    MODELS_DIR, RESULTS_DIR, SEED,
)
from src.data_loader import get_tokenizer, load_from_csv


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        labs  = batch["label"].to(device)
        optimizer.zero_grad()
        out   = model(input_ids=ids, attention_mask=mask, labels=labs)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
        all_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro"))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["label"].to(device)
        out  = model(input_ids=ids, attention_mask=mask, labels=labs)
        total_loss += out.loss.item()
        all_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro"),
            all_preds, all_labels)


def plot_history(history, mode, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes,
                               ["loss", "acc", "f1"],
                               ["Loss", "Accuracy", "Macro-F1"]):
        ax.plot(history[f"train_{key}"], label="train", marker="o")
        ax.plot(history[f"val_{key}"],   label="val",   marker="s")
        ax.set_title(f"{MODEL_SHORT_NAME} ({mode}) — {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Curve saved → {save_path}")


def plot_cm(y_true, y_pred, mode, save_path):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {MODEL_SHORT_NAME} {mode}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Confusion matrix saved → {save_path}")


def train(mode: str):
    """
    mode: 'formal' or 'informal'
    """
    assert mode in ('formal', 'informal'), "mode must be 'formal' or 'informal'"
    set_seed()

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = FORMAL_PROCESSED if mode == 'formal' else INFORMAL_PROCESSED
    tag           = f"{MODEL_SHORT_NAME}_{mode}"

    print(f"\n{'='*55}")
    print(f"  Training: {tag}")
    print(f"  Device  : {device}")
    print(f"{'='*55}")

    tokenizer = get_tokenizer()
    loaders   = load_from_csv(processed_dir, tokenizer, BATCH_SIZE)
    model     = build_model().to(device)

    total_steps  = len(loaders['train']) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    history      = {k: [] for k in
                    ["train_loss","train_acc","train_f1",
                     "val_loss","val_acc","val_f1"]}
    best_val_f1  = 0.0
    patience_ctr = 0
    patience     = 2
    model_path   = os.path.join(MODELS_DIR, f"best_{tag}")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──────────────────────────")
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, loaders['train'], optimizer, scheduler, device)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(
            model, loaders['val'], device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        print(f"  Train → loss:{tr_loss:.4f} acc:{tr_acc:.4f} f1:{tr_f1:.4f}")
        print(f"  Val   → loss:{vl_loss:.4f} acc:{vl_acc:.4f} f1:{vl_f1:.4f}")

        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            patience_ctr = 0
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"  ✓ Best model saved (val F1={best_val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("[train] Early stopping.")
                break

    # Test evaluation
    print(f"\n── Test Evaluation ({tag}) ──────────────────────")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        model_path).to(device)
    _, ts_acc, ts_f1, y_pred, y_true = evaluate(
        best_model, loaders['test'], device)

    label_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    report = classification_report(y_true, y_pred, target_names=label_names)
    print(report)
    print(f"  Test Accuracy : {ts_acc:.4f}")
    print(f"  Test Macro-F1 : {ts_f1:.4f}")

    # Save results
    with open(os.path.join(RESULTS_DIR, f"report_{tag}.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(RESULTS_DIR, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_history(history, mode,
                 os.path.join(RESULTS_DIR, f"training_curve_{tag}.png"))
    plot_cm(y_true, y_pred, mode,
            os.path.join(RESULTS_DIR, f"confusion_matrix_{tag}.png"))

    return ts_acc, ts_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["formal","informal"], required=True,
                        help="Train on formal or informal dataset")
    args = parser.parse_args()
    train(args.mode)
