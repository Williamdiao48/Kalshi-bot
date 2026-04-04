"""Fine-tune DistilBERT on the Kalshi market-article classification dataset.

Reads data/training_set.jsonl (produced by build_training_set.py) and fine-tunes
distilbert-base-uncased to predict market direction (YES=1 / NO=0) from the
concatenated [MARKET]...[SEP]... input text.

Hardware: automatically uses MPS (Apple Silicon) if available, else CPU.

Usage:
    venv/bin/python scripts/train_model.py               # full run
    venv/bin/python scripts/train_model.py --epochs 1    # smoke test

Output:
    models/kalshi_classifier/          — model weights + tokenizer
    models/kalshi_classifier/metrics.json  — val/test accuracy, F1, AUC

Environment variables:
    TRAIN_BATCH_SIZE     Mini-batch size (default: 16)
    TRAIN_LR             Peak learning rate (default: 3e-5)
    TRAIN_EPOCHS         Max epochs (default: 10)
    TRAIN_PATIENCE       Early-stop patience in epochs (default: 2)
    TRAIN_WARMUP_FRAC    Fraction of steps for LR warmup (default: 0.1)
    TRAIN_MAX_LEN        Tokenizer max sequence length (default: 512)
    TRAIN_SEED           Random seed (default: 42)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_root         = Path(__file__).parent.parent
TRAINING_SET  = Path(os.environ.get("TRAINING_SET", _root / "data" / "training_set.jsonl"))
OUTPUT_DIR    = Path(os.environ.get("OUTPUT_DIR",   _root / "models" / "kalshi_classifier"))
MODEL_NAME    = "distilbert-base-uncased"

BATCH_SIZE    = int(os.environ.get("TRAIN_BATCH_SIZE",  "16"))
LR            = float(os.environ.get("TRAIN_LR",        "3e-5"))
MAX_EPOCHS    = int(os.environ.get("TRAIN_EPOCHS",      "10"))
PATIENCE      = int(os.environ.get("TRAIN_PATIENCE",    "2"))
WARMUP_FRAC   = float(os.environ.get("TRAIN_WARMUP_FRAC", "0.1"))
MAX_LEN       = int(os.environ.get("TRAIN_MAX_LEN",     "512"))
SEED          = int(os.environ.get("TRAIN_SEED",        "42"))
FREEZE_BASE   = os.environ.get("TRAIN_FREEZE_BASE", "false").lower() == "true"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer) -> None:
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["input_text"],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(rec["training_label"], dtype=torch.long),
            "sample_weight":  torch.tensor(rec["sample_weight"],  dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    train: bool,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Returns (mean_loss, accuracy)."""
    model.train(train)
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            weights        = batch["sample_weight"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits  # (B, 2)

            # Class-weighted + sample-weighted cross-entropy
            cw = class_weights.to(device) if class_weights is not None else None
            loss_fn    = nn.CrossEntropyLoss(weight=cw, reduction="none")
            per_sample = loss_fn(logits, labels)
            loss       = (per_sample * weights).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * len(labels)
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Returns accuracy, F1, AUC-ROC."""
    model.eval()
    all_labels = []
    all_probs  = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]

        outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
        probs    = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu()

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    preds    = [1 if p >= 0.5 else 0 for p in all_probs]
    accuracy = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, preds, average="binary")
    auc      = roc_auc_score(all_labels, all_probs)

    return {"accuracy": round(accuracy, 4), "f1": round(f1, 4), "auc": round(auc, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(max_epochs: int) -> None:
    _seed_everything(SEED)
    device = _get_device()
    logging.info("Device: %s", device)

    # ---- Load data ---------------------------------------------------------
    logging.info("Loading training set from %s…", TRAINING_SET)
    train_recs, val_recs, test_recs = [], [], []
    for line in TRAINING_SET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        split = rec.get("split", "train")
        if split == "train":
            train_recs.append(rec)
        elif split == "val":
            val_recs.append(rec)
        else:
            test_recs.append(rec)

    logging.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_recs), len(val_recs), len(test_recs),
    )

    # ---- Class weights (handles YES/NO imbalance) ---------------------------
    n_neg = sum(1 for r in train_recs if r["training_label"] == 0)
    n_pos = sum(1 for r in train_recs if r["training_label"] == 1)
    total_train = n_neg + n_pos
    w_neg = total_train / (2 * max(n_neg, 1))
    w_pos = total_train / (2 * max(n_pos, 1))
    logging.info(
        "Class weights — NO: %.3f  YES: %.3f  (n_neg=%d n_pos=%d)",
        w_neg, w_pos, n_neg, n_pos,
    )
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)

    # ---- Tokenizer + model -------------------------------------------------
    logging.info("Loading %s…", MODEL_NAME)
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    if FREEZE_BASE:
        # Freeze the transformer backbone — only train the classification head.
        # Prevents overfitting on small datasets: 590K trainable params instead of 67M.
        for name, param in model.named_parameters():
            if not any(k in name for k in ("pre_classifier", "classifier")):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("Backbone frozen. Trainable params: %d", trainable)

    model = model.to(device)

    # ---- DataLoaders -------------------------------------------------------
    train_ds = PairDataset(train_recs, tokenizer)
    val_ds   = PairDataset(val_recs,   tokenizer)
    test_ds  = PairDataset(test_recs,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Optimizer + scheduler ---------------------------------------------
    total_steps  = len(train_loader) * max_epochs
    warmup_steps = math.ceil(total_steps * WARMUP_FRAC)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop -----------------------------------------------------
    best_val_loss  = float("inf")
    patience_count = 0
    best_epoch     = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, optimizer, scheduler, device, train=True,
            class_weights=class_weights,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, optimizer, scheduler, device, train=False,
            class_weights=None,  # unweighted — honest early-stop signal
        )
        val_metrics = _evaluate_full(model, val_loader, device)

        logging.info(
            "Epoch %2d/%d  train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f "
            "f1=%.3f auc=%.3f",
            epoch, max_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            val_metrics["f1"], val_metrics["auc"],
        )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            # Save best checkpoint
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logging.info("  ✓ New best — checkpoint saved.")
        else:
            patience_count += 1
            logging.info(
                "  No improvement (%d/%d patience).", patience_count, PATIENCE
            )
            if patience_count >= PATIENCE:
                logging.info("Early stopping at epoch %d.", epoch)
                break

    # ---- Final test evaluation ---------------------------------------------
    logging.info("Loading best checkpoint from epoch %d…", best_epoch)
    model = DistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR)
    model = model.to(device)

    val_metrics  = _evaluate_full(model, val_loader,  device)
    test_metrics = _evaluate_full(model, test_loader, device)

    logging.info("=" * 60)
    logging.info(
        "Val  — accuracy=%.4f  f1=%.4f  auc=%.4f",
        val_metrics["accuracy"], val_metrics["f1"], val_metrics["auc"],
    )
    logging.info(
        "Test — accuracy=%.4f  f1=%.4f  auc=%.4f",
        test_metrics["accuracy"], test_metrics["f1"], test_metrics["auc"],
    )

    metrics = {
        "model":       MODEL_NAME,
        "best_epoch":  best_epoch,
        "val":         val_metrics,
        "test":        test_metrics,
        "config": {
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "max_epochs":   max_epochs,
            "patience":     PATIENCE,
            "max_len":      MAX_LEN,
            "seed":         SEED,
            "device":       str(device),
        },
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logging.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Kalshi market pairs.")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    main(max_epochs=args.epochs)
