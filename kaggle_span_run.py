"""Training script for span-based NER (SciBERT + span head).

Usage: run this notebook-style on Kaggle alongside kaggle_run.py.
Results are saved in the same JSON format as the BIO pipeline.
"""
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import ENTITY_TYPES
from src.span_data import (
    NUM_SPAN_CLASSES,
    build_span_datasets,
    compute_span_class_weights,
    span_collate_fn,
)
from src.span_model import SciBertSpan

ROOT        = Path("/kaggle/working")
DATA_DIR    = Path("/kaggle/input/scierc/data")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SpanConfig:
    model_name:        str   = "scibert_span"
    base_model:        str   = "allenai/scibert_scivocab_cased"
    lr:                float = 2e-5
    batch_size:        int   = 16
    num_epochs:        int   = 10
    patience:          int   = 3
    max_length:        int   = 512
    seed:              int   = 42
    augment:           bool  = True
    context_window:    int   = 1
    use_class_weights: bool  = True


# ── evaluation ─────────────────────────────────────────────────────────────

def _collect_spans(
    logits: torch.Tensor,      # [B, S, C]
    span_starts: torch.Tensor, # [B, S]
    span_ends: torch.Tensor,   # [B, S]
    span_labels: torch.Tensor, # [B, S]
    span_mask: torch.Tensor,   # [B, S]
    gold: bool,
) -> list[list[tuple[int, int, int]]]:
    B     = logits.size(0)
    preds = logits.argmax(-1) if not gold else span_labels
    out   = []
    for b in range(B):
        spans = []
        for s in range(span_mask.size(1)):
            if not span_mask[b, s]:
                break
            cls = preds[b, s].item()
            if cls != 0:
                spans.append((span_starts[b, s].item(), span_ends[b, s].item(), cls))
        out.append(spans)
    return out


def evaluate_spans(model, loader, device) -> dict[str, float]:
    model.eval()
    tp = [0] * NUM_SPAN_CLASSES
    fp = [0] * NUM_SPAN_CLASSES
    fn = [0] * NUM_SPAN_CLASSES

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                span_starts=batch["span_starts"],
                span_ends=batch["span_ends"],
            )
            pred_spans = _collect_spans(logits, batch["span_starts"], batch["span_ends"],
                                        batch["span_labels"], batch["span_mask"], gold=False)
            gold_spans = _collect_spans(logits, batch["span_starts"], batch["span_ends"],
                                        batch["span_labels"], batch["span_mask"], gold=True)
            for preds, golds in zip(pred_spans, gold_spans):
                pred_set, gold_set = set(preds), set(golds)
                for s in pred_set:
                    (tp if s in gold_set else fp)[s[2]] += 1
                for s in gold_set:
                    if s not in pred_set:
                        fn[s[2]] += 1

    metrics: dict[str, float] = {}
    f1s = []
    for i, etype in enumerate(ENTITY_TYPES, start=1):
        p  = tp[i] / max(tp[i] + fp[i], 1)
        r  = tp[i] / max(tp[i] + fn[i], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        metrics[f"{etype}_f1"] = round(f1, 4)
        f1s.append(f1)
    metrics["macro_f1"] = round(sum(f1s) / len(f1s), 4)
    return metrics


# ── training ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with autocast():
            loss, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                span_starts=batch["span_starts"],
                span_ends=batch["span_ends"],
                span_labels=batch["span_labels"],
                span_mask=batch["span_mask"],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        n     += 1
    return total / n


def main():
    cfg = SpanConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building span datasets (augment={cfg.augment}, context_window={cfg.context_window})")
    train_ds, dev_ds, test_ds = build_span_datasets(
        DATA_DIR, cfg.base_model, cfg.max_length,
        augment=cfg.augment, context_window=cfg.context_window,
    )
    print(f"  train={len(train_ds)}, dev={len(dev_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=span_collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=span_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=span_collate_fn)

    weights = compute_span_class_weights(train_ds).to(device) if cfg.use_class_weights else None
    model   = SciBertSpan(cfg.base_model, weights)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer   = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=len(train_loader), num_training_steps=total_steps
    )
    scaler = GradScaler()

    best_f1, best_epoch, no_improve = 0.0, 0, 0
    best_state = None
    history    = []

    for epoch in range(1, cfg.num_epochs + 1):
        loss      = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        dev_m     = evaluate_spans(model, dev_loader, device)
        dev_f1    = dev_m["macro_f1"]
        print(f"epoch {epoch:2d}  loss={loss:.4f}  dev_f1={dev_f1:.4f}")
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "dev_macro_f1": dev_f1})

        if dev_f1 > best_f1:
            best_f1, best_epoch, no_improve = dev_f1, epoch, 0
            inner = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in inner.state_dict().items()}
        else:
            no_improve += 1
            if cfg.patience > 0 and no_improve >= cfg.patience:
                print(f"early stop (best={best_f1:.4f} @ epoch {best_epoch})")
                break

    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.load_state_dict(best_state)
    test_m = evaluate_spans(model, test_loader, device)

    print(f"\nTest macro F1: {test_m['macro_f1']:.4f}")
    for et in ENTITY_TYPES:
        print(f"  {et}: {test_m[f'{et}_f1']:.4f}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model_name":   cfg.model_name,
        "run_id":       run_id,
        "config":       asdict(cfg),
        "test_metrics": test_m,
        "history":      history,
    }
    out = RESULTS_DIR / f"{cfg.model_name}_{run_id}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
