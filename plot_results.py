"""
Build all result plots for a given results directory.

Usage:
    python plot_results.py                          # reads results/v1, writes plots/v1
    python plot_results.py --results-dir results/v2
    python plot_results.py --all                    # include macro_f1 < 0.1 runs
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

ENTITY_TYPES = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]
MODEL_ORDER  = [
    "bert_linear_noweight", "bert_linear",
    "roberta_linear_noweight", "roberta_linear",
    "scibert_linear_noweight", "scibert_linear",
    "scibert_mlp",
    "scibert_linear_crf_noweight", "scibert_linear_crf",
    "scibert_concat4",
]
LABELS = {
    "bert_linear_noweight":         "BERT Linear\n(no weight)",
    "bert_linear":                  "BERT Linear\n(weighted)",
    "roberta_linear_noweight":      "RoBERTa Linear\n(no weight)",
    "roberta_linear":               "RoBERTa Linear\n(weighted)",
    "scibert_linear_noweight":      "SciBERT Linear\n(no weight)",
    "scibert_linear":               "SciBERT Linear\n(weighted)",
    "scibert_mlp":                  "SciBERT MLP",
    "scibert_linear_crf_noweight":  "SciBERT CRF\n(no weight)",
    "scibert_linear_crf":           "SciBERT CRF\n(weighted)",
    "scibert_concat4":              "SciBERT Concat-4",
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_csvs(results_dir: Path, include_broken: bool):
    registry = pd.read_csv(results_dir / "registry.csv")
    history  = pd.read_csv(results_dir / "history.csv")

    registry = registry.sort_values("run_id").groupby("model_name", as_index=False).last()
    history  = history.sort_values("run_id").groupby(
        ["model_name", "epoch"], as_index=False
    ).last()

    if not include_broken:
        registry = registry[registry["macro_f1"] >= 0.1]

    models = [m for m in MODEL_ORDER if m in registry["model_name"].values]
    return registry.set_index("model_name"), history, models


def load_predictions(results_dir: Path, model_name: str, run_id: str):
    pattern = f"{model_name}_{run_id}*.json"
    candidates = sorted(results_dir.glob(pattern))
    if not candidates:
        return None, None
    with open(candidates[0]) as f:
        d = json.load(f)
    tp = d.get("test_predictions", {})
    return tp.get("preds"), tp.get("true")


# ── confusion matrix helpers ───────────────────────────────────────────────────

def _extract_spans(bio_seq):
    spans = {}
    i = 0
    while i < len(bio_seq):
        tag = bio_seq[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(bio_seq) and bio_seq[j] == f"I-{etype}":
                j += 1
            spans[(i, j - 1)] = etype
            i = j
        else:
            i += 1
    return spans


def build_entity_confusion(all_preds, all_trues):
    """
    Returns (n+1) x (n+1) count matrix.
    Rows = true type (0..n-1) + "Spurious" (n, FP row).
    Cols = pred type (0..n-1) + "Missed"   (n, FN col).

    Matching: overlap-based. A true span and a pred span are matched if
    they share at least one token position. Each true span is matched to
    the overlapping pred span with the largest overlap (greedy). Unmatched
    true spans → Missed; unmatched pred spans → Spurious.
    """
    n = len(ENTITY_TYPES)
    t2i = {t: i for i, t in enumerate(ENTITY_TYPES)}
    matrix = np.zeros((n + 1, n + 1), dtype=int)

    for pred_seq, true_seq in zip(all_preds, all_trues):
        true_spans = [(s, e, t) for (s, e), t in _extract_spans(true_seq).items()]
        pred_spans = [(s, e, t) for (s, e), t in _extract_spans(pred_seq).items()]

        matched_pred = set()
        for ts, te, ttype in true_spans:
            true_set = set(range(ts, te + 1))
            best, best_overlap = None, 0
            for pi, (ps, pe, ptype) in enumerate(pred_spans):
                if pi in matched_pred:
                    continue
                overlap = len(true_set & set(range(ps, pe + 1)))
                if overlap > best_overlap:
                    best_overlap, best = overlap, pi
            if best is not None:
                ps, pe, ptype = pred_spans[best]
                matched_pred.add(best)
                matrix[t2i[ttype]][t2i[ptype]] += 1
            else:
                matrix[t2i[ttype]][n] += 1  # Missed

        for pi, (ps, pe, ptype) in enumerate(pred_spans):
            if pi not in matched_pred:
                matrix[n][t2i[ptype]] += 1  # Spurious

    return matrix


# ── individual plots ───────────────────────────────────────────────────────────

def plot_macro_f1(registry, models, colors, plots_dir: Path):
    vals   = [registry.loc[m, "macro_f1"] for m in models]
    labels = [LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(13, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Macro F1 (test)", fontsize=11)
    ax.set_title("SciERC NER — Macro F1 by Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(vals) * 1.15 + 0.02))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, plots_dir / "macro_f1_comparison.png")


def plot_entity_f1(registry, models, colors, plots_dir: Path):
    n_models, n_ent = len(models), len(ENTITY_TYPES)
    x, width = np.arange(n_ent), 0.75 / n_models

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, model in enumerate(models):
        vals   = [registry.loc[model, f"{e}_f1"] for e in ENTITY_TYPES]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=LABELS.get(model, model).replace("\n", " "),
               color=colors[i], edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_TYPES, fontsize=10)
    ax.set_ylabel("F1 (test)", fontsize=11)
    ax.set_title("SciERC NER — Per-Entity F1 by Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, plots_dir / "entity_f1_comparison.png")


def plot_heatmap(registry, models, plots_dir: Path):
    data = np.array([[registry.loc[m, f"{e}_f1"] for e in ENTITY_TYPES] for m in models])
    row_labels = [LABELS.get(m, m).replace("\n", " ") for m in models]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(models) + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.82)
    plt.colorbar(im, ax=ax, label="F1")
    ax.set_xticks(range(len(ENTITY_TYPES)))
    ax.set_xticklabels(ENTITY_TYPES, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(len(models)):
        for j in range(len(ENTITY_TYPES)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8.5, color="black", fontweight="bold")
    ax.set_title("SciERC NER — Per-Entity F1 Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, plots_dir / "entity_f1_heatmap.png")


def plot_curves(history, models, colors, plots_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, model in enumerate(models):
        df = history[history["model_name"] == model].sort_values("epoch")
        if df.empty:
            continue
        label = LABELS.get(model, model).replace("\n", " ")
        axes[0].plot(df["epoch"], df["dev_macro_f1"], marker="o", markersize=3,
                     linewidth=1.8, label=label, color=colors[i])
        axes[1].plot(df["epoch"], df["train_loss"],   marker="o", markersize=3,
                     linewidth=1.8, label=label, color=colors[i])
    for ax, title, ylabel, loc in [
        (axes[0], "Dev Macro F1", "Macro F1", "lower right"),
        (axes[1], "Train Loss",   "Loss",      "upper right"),
    ]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc=loc, fontsize=7.5)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("SciERC NER — Learning Curves", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, plots_dir / "learning_curves.png")


def plot_confusion_matrix(matrix: np.ndarray, model_name: str, out_path: Path):
    n = len(ENTITY_TYPES)
    row_labels = ENTITY_TYPES + ["Spurious"]
    col_labels = ENTITY_TYPES + ["Missed"]

    # row-normalize (each true type sums to 1, skip Spurious row for norm)
    norm = matrix.astype(float).copy()
    row_sums = norm.sum(axis=1, keepdims=True).clip(min=1)
    norm = norm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n + 1))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    for i in range(n + 1):
        for j in range(n + 1):
            count = matrix[i, j]
            if count == 0:
                continue
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, str(count), ha="center", va="center",
                    fontsize=8, color=color)

    label = LABELS.get(model_name, model_name).replace("\n", " ")
    ax.set_title(f"Entity Confusion — {label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ── precision / recall ────────────────────────────────────────────────────────

def compute_pr(preds: list, trues: list) -> dict[str, dict[str, float]]:
    """Compute per-entity precision, recall, F1 from BIO sequences via seqeval."""
    report = classification_report(trues, preds, output_dict=True, zero_division=0)
    result = {}
    for etype in ENTITY_TYPES:
        r = report.get(etype, {})
        result[etype] = {
            "precision": r.get("precision", 0.0),
            "recall":    r.get("recall",    0.0),
            "f1":        r.get("f1-score",  0.0),
        }
    return result


def plot_pr_bars(pr: dict, model_name: str, plots_dir: Path):
    """P / R / F1 bar chart for a single model."""
    x      = np.arange(len(ENTITY_TYPES))
    width  = 0.25
    ps     = [pr[e]["precision"] for e in ENTITY_TYPES]
    rs     = [pr[e]["recall"]    for e in ENTITY_TYPES]
    fs     = [pr[e]["f1"]        for e in ENTITY_TYPES]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - width, ps, width, label="Precision", color="#4C72B0", zorder=3)
    ax.bar(x,         rs, width, label="Recall",    color="#DD8452", zorder=3)
    ax.bar(x + width, fs, width, label="F1",        color="#55A868", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_TYPES, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    label = LABELS.get(model_name, model_name).replace("\n", " ")
    ax.set_title(f"Precision / Recall / F1 — {label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, plots_dir / f"pr_bars_{model_name}.png")


def plot_pr_scatter(pr_all: dict[str, dict], models: list[str], plots_dir: Path):
    """Precision vs Recall scatter — one point per (model, entity_type)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^", "D", "v", "P"]
    colors  = [plt.cm.tab10(i / 10) for i in range(len(models))]

    for i, model in enumerate(models):
        if model not in pr_all:
            continue
        pr = pr_all[model]
        for j, etype in enumerate(ENTITY_TYPES):
            p, r = pr[etype]["precision"], pr[etype]["recall"]
            ax.scatter(r, p, color=colors[i], marker=markers[j % len(markers)],
                       s=70, zorder=3, alpha=0.85)

    # Legend: models by color
    for i, model in enumerate(models):
        ax.scatter([], [], color=colors[i], s=50,
                   label=LABELS.get(model, model).replace("\n", " "))
    # Legend: entity types by marker
    for j, etype in enumerate(ENTITY_TYPES):
        ax.scatter([], [], marker=markers[j % len(markers)], color="gray", s=50, label=etype)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3)  # P=R diagonal
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision vs Recall per Entity Type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(linestyle="--", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, plots_dir / "pr_scatter.png")


def plot_pr_heatmaps(pr_all: dict[str, dict], models: list[str], plots_dir: Path):
    """Side-by-side heatmaps: precision (left) and recall (right)."""
    p_data = np.array([[pr_all[m][e]["precision"] for e in ENTITY_TYPES]
                        for m in models if m in pr_all])
    r_data = np.array([[pr_all[m][e]["recall"]    for e in ENTITY_TYPES]
                        for m in models if m in pr_all])
    row_labels = [LABELS.get(m, m).replace("\n", " ") for m in models if m in pr_all]

    fig, axes = plt.subplots(1, 2, figsize=(16, 0.55 * len(row_labels) + 1.5))
    for ax, data, title in zip(axes, [p_data, r_data], ["Precision", "Recall"]):
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.9)
        plt.colorbar(im, ax=ax, label=title)
        ax.set_xticks(range(len(ENTITY_TYPES)))
        ax.set_xticklabels(ENTITY_TYPES, fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        for i in range(len(row_labels)):
            for j in range(len(ENTITY_TYPES)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")
        ax.set_title(f"{title} per Entity Type", fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, plots_dir / "pr_heatmaps.png")


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/v1")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir   = Path("plots") / results_dir.name
    conf_dir    = plots_dir / "conf_mtx"
    conf_dir.mkdir(parents=True, exist_ok=True)

    registry, history, models = load_csvs(results_dir, include_broken=args.all)
    print(f"[{results_dir.name}] {len(models)} models → {plots_dir}\n")

    colors = [plt.cm.tab10(i / 10) for i in range(len(models))]

    print("summary plots")
    plot_macro_f1(registry, models, colors, plots_dir)
    plot_entity_f1(registry, models, colors, plots_dir)
    plot_heatmap(registry, models, plots_dir)
    plot_curves(history, models, colors, plots_dir)

    print("\nconfusion matrices + precision/recall")
    pr_all = {}
    for model in models:
        run_id = registry.loc[model, "run_id"]
        preds, trues = load_predictions(results_dir, model, run_id)
        if not preds:
            print(f"  skip {model} — no predictions")
            continue
        matrix = build_entity_confusion(preds, trues)
        plot_confusion_matrix(matrix, model, conf_dir / f"{model}.png")
        pr_all[model] = compute_pr(preds, trues)

    if pr_all:
        print("\nprecision/recall plots")
        top5 = sorted(pr_all, key=lambda m: registry.loc[m, "macro_f1"], reverse=True)[:5]
        best = top5[0]
        plot_pr_bars(pr_all[best], best, plots_dir)
        plot_pr_scatter(pr_all, top5, plots_dir)
        plot_pr_heatmaps(pr_all, top5, plots_dir)


if __name__ == "__main__":
    main()
