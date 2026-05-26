"""
Build comparison plots from results/registry.csv and results/history.csv.
Skips models with macro_f1 < 0.1 (broken runs).

Usage:
    python plot_results.py
    python plot_results.py --all   # include broken runs too
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR  = Path("results")
PLOTS_DIR    = Path("plots")
ENTITY_TYPES = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]
MODEL_ORDER  = ["bert_linear", "scibert_linear", "scibert_mlp",
                "scibert_crf", "scibert_mlp_crf", "scibert_concat4", "deberta_qlora"]
LABELS = {
    "bert_linear":     "BERT\nLinear",
    "scibert_linear":  "SciBERT\nLinear",
    "scibert_mlp":     "SciBERT\nMLP",
    "scibert_crf":     "SciBERT\nCRF",
    "scibert_mlp_crf": "SciBERT\nMLP+CRF",
    "scibert_concat4": "SciBERT\nConcat-4",
    "deberta_qlora":   "DeBERTa\nQLoRA",
}


def load_data(include_broken: bool):
    registry = pd.read_csv(RESULTS_DIR / "registry.csv")
    history  = pd.read_csv(RESULTS_DIR / "history.csv")

    # latest run per model
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


def plot_macro_f1(registry, models, colors):
    vals   = [registry.loc[m, "macro_f1"] for m in models]
    labels = [LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4))
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
    out = PLOTS_DIR / "macro_f1_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"saved → {out}")
    plt.show()


def plot_entity_f1(registry, models, colors):
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
    out = PLOTS_DIR / "entity_f1_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"saved → {out}")
    plt.show()


def plot_curves(history, models, colors):
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
    out = PLOTS_DIR / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved → {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="include broken runs (macro_f1 < 0.1)")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(exist_ok=True)
    registry, history, models = load_data(include_broken=args.all)
    print(f"Plotting: {models}\n")

    colors = palette(len(models))
    plot_macro_f1(registry, models, colors)
    plot_entity_f1(registry, models, colors)
    plot_curves(history, models, colors)


if __name__ == "__main__":
    main()
