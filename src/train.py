"""
train.py
────────
Fine-tune bert-base-multilingual-cased (mBERT) for 3-class sentiment
analysis on Indonesian PPKM tweets.

Usage:
    python -m src.train
"""

import os
import random
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
import json

from src.config import (
    PRETRAINED_MODEL, MODEL_SHORT_NAME, NUM_LABELS, LABEL_MAP,
    EPOCHS, LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY, DROPOUT,
    BATCH_SIZE, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, SEED,
)
from src.preprocessing import load_and_clean, split_dataset
from src.data_loader import get_tokenizer, make_dataloaders


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Model ───────────────────────────────────────────────────────────────────

def build_model(num_labels: int = NUM_LABELS, dropout: float = DROPOUT):
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    return model


# ─── Training & evaluation helpers ───────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, all_preds, all_labels


# ─── Visualisation helpers ───────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("loss", "Loss"), ("acc", "Accuracy"), ("f1", "Macro-F1")]

    for ax, (key, title) in zip(axes, metrics):
        ax.plot(history[f"train_{key}"], label="train", marker="o")
        ax.plot(history[f"val_{key}"],   label="val",   marker="s")
        ax.set_title(f"{MODEL_SHORT_NAME} — {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Training curve saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {MODEL_SHORT_NAME} (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Confusion matrix saved → {save_path}")


# ─── Main training loop ───────────────────────────────────────────────────────

def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device  : {device}")
    print(f"[train] Model         : {PRETRAINED_MODEL}  ({MODEL_SHORT_NAME})")

    # 1. Data
    df     = load_and_clean()
    splits = split_dataset(df)

    tokenizer = get_tokenizer()
    loaders   = make_dataloaders(splits, tokenizer, BATCH_SIZE)

    # 2. Model
    model = build_model().to(device)

    # 3. Optimiser & scheduler
    total_steps  = len(loaders["train"]) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 4. Training loop with early stopping
    history       = {k: [] for k in
                     ["train_loss", "train_acc", "train_f1",
                      "val_loss",   "val_acc",   "val_f1"]}
    best_val_f1   = 0.0
    patience      = 2
    patience_ctr  = 0
    best_model_path = os.path.join(MODELS_DIR, f"best_{MODEL_SHORT_NAME}_model")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} {'─'*40}")

        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, loaders["train"], optimizer, scheduler, device)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(
            model, loaders["val"], device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        print(f"  Train → loss: {tr_loss:.4f} | acc: {tr_acc:.4f} | f1: {tr_f1:.4f}")
        print(f"  Val   → loss: {vl_loss:.4f} | acc: {vl_acc:.4f} | f1: {vl_f1:.4f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            patience_ctr = 0
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ Best model saved (val F1 = {best_val_f1:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                print("[train] Early stopping triggered.")
                break

    # 5. Test evaluation (load best model)
    print(f"\n── Test Evaluation ({MODEL_SHORT_NAME}) {'─'*35}")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path).to(device)
    _, ts_acc, ts_f1, y_pred, y_true = evaluate(
        best_model, loaders["test"], device)

    report = classification_report(
        y_true, y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    )
    print(report)
    print(f"  Test Accuracy : {ts_acc:.4f}")
    print(f"  Test Macro-F1 : {ts_f1:.4f}")

    # 6. Save results  (filenames include model short name)
    report_path = os.path.join(RESULTS_DIR,
                               f"classification_report_{MODEL_SHORT_NAME}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    history_path = os.path.join(RESULTS_DIR, f"history_{MODEL_SHORT_NAME}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_training_history(
        history,
        os.path.join(RESULTS_DIR, f"training_curve_{MODEL_SHORT_NAME}.png"))
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(RESULTS_DIR, f"confusion_matrix_{MODEL_SHORT_NAME}.png"))

    print(f"\n[train] All results saved to: {RESULTS_DIR}")
    return history, y_true, y_pred


if __name__ == "__main__":
    train()
