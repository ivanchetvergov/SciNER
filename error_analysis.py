"""
Error analysis for trained NER models.
Reads result JSON files from results/ and generates:
  - Entity confusion heatmap  → plots/confusion_{model}.png
  - Span-length F1 chart      → plots/span_length_f1.png
  - Top error sentences       → stdout

Usage:
    python error_analysis.py                  # best model by macro F1
    python error_analysis.py --model scibert_crf
    python error_analysis.py --all            # all models
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR  = Path("results")
PLOTS_DIR    = Path("plots")
ENTITY_TYPES = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]


# ── span utilities ─────────────────────────────────────────────────────────────

def extract_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    spans, start, etype = [], None, None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1, etype))
            start, etype = i, tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == etype:
            pass
        else:
            if start is not None:
                spans.append((start, i - 1, etype))
            start, etype = None, None
    if start is not None:
        spans.append((start, len(tags) - 1, etype))
    return spans


def span_confusion(all_preds, all_true):
    """Returns (matrix, fp_counts).
    matrix[true_type][pred_type | "MISS"] = count
    fp_counts[pred_type] = spans predicted with no matching true span
    """
    cols = ENTITY_TYPES + ["MISS"]
    matrix = {t: {c: 0 for c in cols} for t in ENTITY_TYPES}
    fp_counts = {t: 0 for t in ENTITY_TYPES}

    for pred_seq, true_seq in zip(all_preds, all_true):
        true_spans = {(s, e): t for s, e, t in extract_spans(true_seq)}
        pred_spans = {(s, e): t for s, e, t in extract_spans(pred_seq)}

        for (s, e), true_type in true_spans.items():
            pred_type = pred_spans.get((s, e), "MISS")
            matrix[true_type][pred_type] += 1

        for (s, e), pred_type in pred_spans.items():
            if (s, e) not in true_spans:
                fp_counts[pred_type] += 1

    return matrix, fp_counts


def f1_by_length(all_preds, all_true) -> dict[int, float]:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred_seq, true_seq in zip(all_preds, all_true):
        true_spans = {(s, e): t for s, e, t in extract_spans(true_seq)}
        pred_spans = {(s, e): t for s, e, t in extract_spans(pred_seq)}

        for (s, e), true_type in true_spans.items():
            length = e - s + 1
            if pred_spans.get((s, e)) == true_type:
                tp[length] += 1
            else:
                fn[length] += 1

        for (s, e), pred_type in pred_spans.items():
            length = e - s + 1
            if true_spans.get((s, e)) != pred_type:
                fp[length] += 1

    results = {}
    for length in sorted(set(tp) | set(fn)):
        p = tp[length] / (tp[length] + fp.get(length, 0) + 1e-9)
        r = tp[length] / (tp[length] + fn[length] + 1e-9)
        results[length] = 2 * p * r / (p + r + 1e-9)
    return results


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_confusion(matrix, fp_counts, model_name):
    cols = ENTITY_TYPES + ["MISS"]
    rows = ENTITY_TYPES + ["FALSE+"]

    data = np.zeros((len(rows), len(cols)))
    for i, row_type in enumerate(ENTITY_TYPES):
        for j, col_type in enumerate(cols):
            data[i, j] = matrix[row_type][col_type]
    for j, col_type in enumerate(ENTITY_TYPES):
        data[len(ENTITY_TYPES), j] = fp_counts[col_type]

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(data, cmap="Blues", aspect="auto")

    short_cols = [c.replace("OtherScientificTerm", "OtherSci") for c in cols]
    short_rows = [r.replace("OtherScientificTerm", "OtherSci") for r in rows]
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(short_cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(short_rows, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Entity Confusion — {model_name}", fontsize=13, fontweight="bold")

    thresh = data.max() * 0.55
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = int(data[i, j])
            if val > 0:
                color = "white" if data[i, j] > thresh else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()
    out = PLOTS_DIR / f"confusion_{model_name}.png"
    plt.savefig(out, dpi=150)
    print(f"saved → {out}")
    plt.close()


def plot_span_length(length_f1_by_model: dict[str, dict]):
    """Bar chart: F1 by span length (1, 2, 3+) for each model."""
    buckets = {"1": 1, "2": 2, "3+": None}
    model_names = list(length_f1_by_model.keys())
    colors = [plt.cm.tab10(i / 10) for i in range(len(model_names))]

    x = np.arange(len(buckets))
    width = 0.7 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, lf1) in enumerate(length_f1_by_model.items()):
        vals = []
        # bucket 1, 2, 3+
        vals.append(lf1.get(1, 0.0))
        vals.append(lf1.get(2, 0.0))
        # 3+ = average of all lengths >= 3, weighted by frequency (approx mean)
        long_vals = [v for k, v in lf1.items() if k >= 3]
        vals.append(np.mean(long_vals) if long_vals else 0.0)

        offset = (i - len(model_names) / 2 + 0.5) * width
        label = model.replace("_", " ")
        ax.bar(x + offset, vals, width * 0.9, label=label, color=colors[i],
               edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(list(buckets.keys()), fontsize=11)
    ax.set_xlabel("Span length (tokens)", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("F1 by Entity Span Length", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "span_length_f1.png"
    plt.savefig(out, dpi=150)
    print(f"saved → {out}")
    plt.close()


# ── error examples ─────────────────────────────────────────────────────────────

def print_error_examples(all_preds, all_true, model_name, n=5):
    scored = []
    for i, (pred_seq, true_seq) in enumerate(zip(all_preds, all_true)):
        true_spans = {(s, e): t for s, e, t in extract_spans(true_seq)}
        pred_spans = {(s, e): t for s, e, t in extract_spans(pred_seq)}
        errors = sum(1 for k, v in true_spans.items() if pred_spans.get(k) != v)
        errors += sum(1 for k in pred_spans if k not in true_spans)
        scored.append((errors, i))

    print(f"\n── Top {n} error sentences [{model_name}] ──")
    for errors, idx in sorted(scored, reverse=True)[:n]:
        pred_seq = all_preds[idx]
        true_seq = all_true[idx]
        true_spans = extract_spans(true_seq)
        pred_spans = extract_spans(pred_seq)
        print(f"\n  sentence #{idx}  ({errors} span errors)")
        print(f"  TRUE: {true_spans}")
        print(f"  PRED: {pred_spans}")


# ── main ───────────────────────────────────────────────────────────────────────

def load_predictions(model_name: str) -> tuple[list, list] | None:
    registry = pd.read_csv(RESULTS_DIR / "registry.csv")
    row = registry[registry["model_name"] == model_name].sort_values("run_id").iloc[-1]
    path = RESULTS_DIR / row["results_file"]
    if not path.exists():
        print(f"[skip] result file not found: {path}")
        return None
    with open(path) as f:
        result = json.load(f)
    if "test_predictions" not in result:
        print(f"[skip] {model_name}: no saved predictions (re-run to generate them)")
        return None
    preds = result["test_predictions"]
    return preds["preds"], preds["true"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="specific model to analyze")
    parser.add_argument("--all", action="store_true", help="analyze all models in registry")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(exist_ok=True)

    registry = pd.read_csv(RESULTS_DIR / "registry.csv")
    registry = registry.sort_values("run_id").groupby("model_name", as_index=False).last()

    if args.model:
        models = [args.model]
    elif args.all:
        models = registry["model_name"].tolist()
    else:
        # default: best model
        best = registry.loc[registry["macro_f1"].idxmax(), "model_name"]
        models = [best]
        print(f"Analyzing best model: {best}  (macro F1 = {registry['macro_f1'].max():.4f})")

    length_f1_by_model = {}

    for model_name in models:
        result = load_predictions(model_name)
        if result is None:
            continue
        all_preds, all_true = result

        matrix, fp_counts = span_confusion(all_preds, all_true)
        plot_confusion(matrix, fp_counts, model_name)

        lf1 = f1_by_length(all_preds, all_true)
        length_f1_by_model[model_name] = lf1

        print_error_examples(all_preds, all_true, model_name)

    if length_f1_by_model:
        plot_span_length(length_f1_by_model)


if __name__ == "__main__":
    main()
