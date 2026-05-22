import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score

ENTITY_TYPES = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]

_REGISTRY_COLS = [
    "run_id", "model_name", "base_model",
    "num_epochs", "batch_size", "lr", "seed",
    "macro_f1",
    *[f"{e}_f1" for e in ENTITY_TYPES],
    "results_file",
]

_HISTORY_COLS = ["run_id", "model_name", "epoch", "train_loss", "dev_macro_f1"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_predictions(
    preds: torch.Tensor,
    labels: torch.Tensor,
    id2label: dict,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert batched logits + label tensors to seqeval-style tag lists."""
    all_preds, all_true = [], []
    pred_ids = preds.argmax(dim=-1)

    for pred_seq, label_seq in zip(pred_ids, labels):
        pred_tags, true_tags = [], []
        for p, l in zip(pred_seq, label_seq):
            if l.item() == -100:
                continue
            pred_tags.append(id2label[p.item()])
            true_tags.append(id2label[l.item()])
        all_preds.append(pred_tags)
        all_true.append(true_tags)

    return all_preds, all_true


def decode_crf_predictions(
    tag_seqs: list[list[int]],
    labels: torch.Tensor,
    id2label: dict,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert CRF decoded tag sequences to seqeval-style tag lists."""
    all_preds, all_true = [], []

    for tags, label_seq in zip(tag_seqs, labels):
        pred_tags, true_tags = [], []
        tag_iter = iter(tags)
        for l in label_seq:
            if l.item() == -100:
                continue
            true_tags.append(id2label[l.item()])
            try:
                pred_tags.append(id2label[next(tag_iter)])
            except StopIteration:
                pred_tags.append("O")
        all_preds.append(pred_tags)
        all_true.append(true_tags)

    return all_preds, all_true


def compute_metrics(
    all_preds: list[list[str]],
    all_true: list[list[str]],
    entity_types: list[str],
) -> dict:
    report = seqeval_report(all_true, all_preds, output_dict=True, zero_division=0)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    metrics = {"macro_f1": macro_f1}
    for etype in entity_types:
        if etype in report:
            metrics[f"{etype}_f1"] = report[etype]["f1-score"]
        else:
            metrics[f"{etype}_f1"] = 0.0

    return metrics


def append_registry(result: dict, registry_path: str | Path) -> None:
    """Append one experiment result as a row in registry.csv."""
    path = Path(registry_path)
    write_header = not path.exists()
    m = result["test_metrics"]
    cfg = result["config"]
    row = {
        "run_id":       result["run_id"],
        "model_name":   result["model_name"],
        "base_model":   result["base_model"],
        "num_epochs":   cfg.get("num_epochs"),
        "batch_size":   cfg.get("batch_size"),
        "lr":           cfg.get("lr"),
        "seed":         cfg.get("seed"),
        "macro_f1":     round(m["macro_f1"], 4),
        **{f"{e}_f1": round(m.get(f"{e}_f1", 0), 4) for e in ENTITY_TYPES},
        "results_file": result.get("results_file", ""),
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_REGISTRY_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_history(result: dict, history_path: str | Path) -> None:
    """Append per-epoch training history rows to history.csv."""
    path = Path(history_path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_COLS)
        if write_header:
            writer.writeheader()
        for h in result["history"]:
            writer.writerow({
                "run_id":       result["run_id"],
                "model_name":   result["model_name"],
                "epoch":        h["epoch"],
                "train_loss":   round(h["train_loss"], 4),
                "dev_macro_f1": round(h["dev_macro_f1"], 4),
            })


def save_results(results: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_checkpoint(model, path: str | Path):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str | Path):
    import torch
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
