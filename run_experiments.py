"""
Run all 6 NER experiments sequentially and save results.
Usage: python run_experiments.py [--data-dir data] [--results-dir results]
"""
import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import EXPERIMENTS
from src.data import build_datasets, ID2LABEL, NUM_LABELS, ENTITY_TYPES
from src.models import build_model
from src.train import train_model, evaluate
from src.utils import set_seed, save_results


def run_experiment(cfg, data_dir: Path, results_dir: Path, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  Experiment: {cfg.model_name}")
    print(f"{'='*60}")

    set_seed(cfg.seed)

    print(f"Loading data for tokenizer: {cfg.base_model}")
    train_ds, dev_ds, test_ds = build_datasets(data_dir, cfg.base_model, cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Building model: {cfg.model_name}")
    model = build_model(
        cfg.model_name, cfg.base_model, NUM_LABELS,
        cfg.use_qlora, cfg.lora_rank, cfg.lora_alpha,
    )
    # QLoRA: model stays on GPU via device_map; others need explicit .to()
    if not cfg.use_qlora:
        model = model.to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0.01,
    )

    _, history = train_model(
        model, train_loader, dev_loader, optimizer, device,
        cfg.num_epochs, cfg.model_name,
    )

    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, ID2LABEL)

    result = {
        "model_name": cfg.model_name,
        "base_model": cfg.base_model,
        "config": cfg.__dict__,
        "history": history,
        "test_metrics": test_metrics,
    }

    save_results(result, results_dir / f"{cfg.model_name}.json")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    return result


def print_summary(all_results: list[dict]):
    COLS = ["model", "macro_F1"] + [f"{e}_F1" for e in ENTITY_TYPES]
    widths = [max(len(c), 20) for c in COLS]
    widths[0] = 20

    header = "  ".join(c.ljust(w) for c, w in zip(COLS, widths))
    print(f"\n{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for res in all_results:
        m = res["test_metrics"]
        row = [
            res["model_name"],
            f"{m['macro_f1']:.4f}",
        ] + [f"{m.get(f'{e}_f1', 0):.4f}" for e in ENTITY_TYPES]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"{'='*len(header)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="data",    help="Path to SciERC JSON files")
    parser.add_argument("--results-dir", default="results", help="Where to write result JSONs")
    parser.add_argument("--models",      nargs="*",          help="Subset of model names to run")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    experiments = EXPERIMENTS
    if args.models:
        experiments = [e for e in experiments if e.model_name in args.models]

    all_results = []
    for cfg in experiments:
        result = run_experiment(cfg, data_dir, results_dir, device)
        all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()
