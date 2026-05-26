"""
Build comparison plots from results/registry.csv and results/history.csv.
Skips models with macro_f1 < 0.1 (broken runs).

Usage:
    python plot_results.py
    python plot_results.py --results-dir results/v1
    python plot_results.py --all   # include broken runs too
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR    = Path("plots")
ENTITY_TYPES = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]
MODEL_ORDER  = [
    "bert_linear_noweight", "bert_linear",
    "roberta_linear_noweight", "roberta_linear",
    "scibert_linear_noweight", "scibert_linear",
    "scibert_mlp",
    "scibert_linear_crf_noweight", "scibert_linear_crf", "scibert_concat4",
]
LABELS = {
    "bert_linear_noweight":         "BERT\nLinear\n(no weight)",
    "bert_linear":                  "BERT\nLinear\n(weighted)",
    "roberta_linear_noweight":      "RoBERTa\nLinear\n(no weight)",
    "roberta_linear":               "RoBERTa\nLinear\n(weighted)",
    "scibert_linear_noweight":      "SciBERT\nLinear\n(no weight)",
    "scibert_linear":               "SciBERT\nLinear\n(weighted)",
    "scibert_mlp":                  "SciBERT\nMLP",
    "scibert_linear_crf_noweight":  "SciBERT\nLinear+CRF\n(no weight)",
    "scibert_linear_crf":           "SciBERT\nLinear+CRF\n(weighted)",
    "scibert_concat4":              "SciBERT\nConcat-4",
}


def load_data(results_dir: Path, include_broken: bool):
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


def palette(n):
    return [plt.cm.tab10(i / 10) for i in range(n)]


def plot_macro_f1(registry, models, colors, plots_dir: Path):
    vals   = [registry.loc[m, "macro_f1"] for m in models]
    labels = [LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(11, 4))
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
    out = plots_dir / "macro_f1_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"saved → {out}")
    plt.close()


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
    out = plots_dir / "entity_f1_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"saved → {out}")
    plt.close()


def plot_curves(history, models, colors, plots_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, model in enumerate(models):
        df = history[history["model_name"] == model].sort_values("epoch")
        if df.empty:
            continue
        label = LABELS.get(model, model).replace("\n", " ")
        axes[0].plot(df["epoch"], df["dev_macro_f1"], marker="o", markersize=4,
                     linewidth=1.8, label=label, color=colors[i])
        axes[1].plot(df["epoch"], df["train_loss"],   marker="o", markersize=4,
                     linewidth=1.8, label=label, color=colors[i])

    for ax, title, ylabel, loc in [
        (axes[0], "Dev Macro F1",  "Macro F1", "lower right"),
        (axes[1], "Train Loss",    "Loss",      "upper right"),
    ]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc=loc, fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("SciERC NER — Learning Curves", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = plots_dir / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved → {out}")
    plt.close()


def plot_heatmap(registry, models, plots_dir: Path):
    data = np.array([[registry.loc[m, f"{e}_f1"] for e in ENTITY_TYPES] for m in models])
    row_labels = [LABELS.get(m, m).replace("\n", " ") for m in models]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(models) + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.80)
    plt.colorbar(im, ax=ax, label="F1")

    ax.set_xticks(range(len(ENTITY_TYPES)))
    ax.set_xticklabels(ENTITY_TYPES, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(models)):
        for j in range(len(ENTITY_TYPES)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_title("SciERC NER — Per-Entity F1 Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "entity_f1_heatmap.png"
    plt.savefig(out, dpi=150)
    print(f"saved → {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="path to results directory")
    parser.add_argument("--plots-dir",   default=None,      help="output plots dir (default: plots/<results-dir-name>)")
    parser.add_argument("--all", action="store_true", help="include broken runs (macro_f1 < 0.1)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir   = Path(args.plots_dir) if args.plots_dir else PLOTS_DIR / results_dir.name
    plots_dir.mkdir(parents=True, exist_ok=True)

    registry, history, models = load_data(results_dir, include_broken=args.all)
    print(f"Results dir : {results_dir}")
    print(f"Plots dir   : {plots_dir}")
    print(f"Models ({len(models)}): {models}\n")

    colors = palette(len(models))
    plot_macro_f1(registry, models, colors, plots_dir)
    plot_entity_f1(registry, models, colors, plots_dir)
    plot_curves(history, models, colors, plots_dir)
    plot_heatmap(registry, models, plots_dir)


if __name__ == "__main__":
    main()
