"""
Entry point for Kaggle.

Usage (in notebook cell):
    !python kaggle_run.py
    !python kaggle_run.py --models bert_linear scibert_linear   # subset
    !python kaggle_run.py --epochs 5 --batch-size 8             # override
"""
import argparse
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime
from math import ceil
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import EXPERIMENTS, ExperimentConfig
from src.data import build_datasets, compute_class_weights, ID2LABEL, NUM_LABELS, ENTITY_TYPES
from src.models import build_model, is_crf_model
from src.train import train_model, evaluate
from src.utils import set_seed, save_results, append_registry, append_history


# ── paths ──────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent
DATA_DIR        = ROOT / "data"
RESULTS_DIR     = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"


# ── data ───────────────────────────────────────────────────────────────────────
def ensure_data():
    if (DATA_DIR / "train.json").exists():
        print(f"[data] already present in {DATA_DIR}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    archive = ROOT / "scierc.tar.gz"
    print("[data] downloading SciERC...")
    urllib.request.urlretrieve(
        "https://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz",
        archive,
    )
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(ROOT)
    for f in (ROOT / "processed_data" / "json").glob("*.json"):
        shutil.copy(f, DATA_DIR / f.name)
    print(f"[data] ready: {[p.name for p in DATA_DIR.glob('*.json')]}")


# ── single experiment ──────────────────────────────────────────────────────────
def run_experiment(cfg: ExperimentConfig, device: torch.device, run_id: str) -> dict:
    print(f"\n{'='*60}\n  {cfg.model_name}  |  {cfg.base_model}\n{'='*60}")
    set_seed(cfg.seed)

    train_ds, dev_ds, test_ds = build_datasets(DATA_DIR, cfg.base_model, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = compute_class_weights(train_ds, NUM_LABELS)

    model = build_model(cfg.model_name, cfg.base_model, NUM_LABELS,
                        cfg.use_qlora, cfg.lora_rank, cfg.lora_alpha,
                        class_weights=class_weights)
    if not cfg.use_qlora:
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"[gpu] using {torch.cuda.device_count()} GPUs")

    if is_crf_model(model) and cfg.crf_lr > 0:
        crf_param_ids = {id(p) for p in model.crf.parameters()}
        optimizer = AdamW([
            {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in crf_param_ids],
             "lr": cfg.lr},
            {"params": [p for p in model.crf.parameters() if p.requires_grad],
             "lr": cfg.crf_lr},
        ], weight_decay=0.01)
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=0.01,
        )

    steps_per_epoch = ceil(len(train_loader) / cfg.grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps_per_epoch,
        num_training_steps=steps_per_epoch * cfg.num_epochs,
    )

    _, history = train_model(model, train_loader, dev_loader, optimizer,
                             device, cfg.num_epochs, cfg.model_name,
                             patience=cfg.early_stopping_patience,
                             scheduler=scheduler,
                             grad_accum_steps=cfg.grad_accum_steps)

    ckpt_path = CHECKPOINTS_DIR / f"{cfg.model_name}.pt"
    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state, ckpt_path)
    print(f"[ckpt] saved → {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.0f} MB)")

    test_metrics, test_preds, test_true = evaluate(
        model, test_loader, device, ID2LABEL, return_preds=True
    )

    results_file = f"{cfg.model_name}_{run_id}.json"
    result = {
        "run_id":       run_id,
        "model_name":   cfg.model_name,
        "base_model":   cfg.base_model,
        "config":       cfg.__dict__,
        "history":      history,
        "test_metrics": test_metrics,
        "test_predictions": {"preds": test_preds, "true": test_true},
        "results_file": results_file,
    }
    save_results(result, RESULTS_DIR / results_file)
    append_registry(result, RESULTS_DIR / "registry.csv")
    append_history(result,   RESULTS_DIR / "history.csv")

    print(f"\n[test] macro F1: {test_metrics['macro_f1']:.4f}")
    for et in ENTITY_TYPES:
        print(f"       {et:<24} {test_metrics.get(et+'_f1', 0):.4f}")

    return result


# ── summary table ──────────────────────────────────────────────────────────────
def print_summary(all_results: list[dict]):
    cols   = ["model", "macro_F1"] + [f"{e}_F1" for e in ENTITY_TYPES]
    widths = [22] + [10] * len(ENTITY_TYPES) + [10]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for res in all_results:
        m = res["test_metrics"]
        row = [res["model_name"], f"{m['macro_f1']:.4f}"]
        row += [f"{m.get(f'{e}_f1', 0):.4f}" for e in ENTITY_TYPES]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(sep)


# ── pack for download ──────────────────────────────────────────────────────────
def pack_artifacts():
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT
    archives = []
    for name, src in [("results", RESULTS_DIR), ("checkpoints", CHECKPOINTS_DIR)]:
        if src.exists() and any(src.iterdir()):
            dest = out / name
            shutil.make_archive(str(dest), "zip", src)
            arc = dest.with_suffix(".zip")
            archives.append((arc.name, arc.stat().st_size / 1e6))

    print("\n[artifacts]")
    for name, mb in archives:
        print(f"  {name:<30} {mb:>8.1f} MB")
    if str(out) == "/kaggle/working":
        print("\nDownload via:  Output → (right panel) → Download")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="*", help="model names to run (default: all)")
    parser.add_argument("--epochs",     type=int,  help="override num_epochs")
    parser.add_argument("--batch-size", type=int,  help="override batch_size")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu]    {torch.cuda.get_device_name(0)}"
              f"  |  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    ensure_data()

    experiments = EXPERIMENTS
    if args.models:
        experiments = [e for e in experiments if e.model_name in args.models]

    if args.epochs or args.batch_size:
        patched = []
        for cfg in experiments:
            from dataclasses import replace
            cfg = replace(cfg,
                          **({} if not args.epochs     else {"num_epochs":  args.epochs}),
                          **({} if not args.batch_size else {"batch_size": args.batch_size}))
            patched.append(cfg)
        experiments = patched

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[run_id] {run_id}")

    all_results = []
    for cfg in experiments:
        result = run_experiment(cfg, device, run_id)
        all_results.append(result)

    print_summary(all_results)
    pack_artifacts()


if __name__ == "__main__":
    main()
