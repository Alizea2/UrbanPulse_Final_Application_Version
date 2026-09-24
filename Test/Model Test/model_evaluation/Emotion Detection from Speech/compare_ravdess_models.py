"""Merges the four RAVDESS evaluations (HuBERT, Yassmen, audEERING, GoEmotions)
into one metrics.txt and one set of comparison graphs.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Tests/model_evaluation/Emotion Detection from Speech
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../Tests
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Emotion Detection from Speech")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "Model_Comparison_RAVDESS")

# (display name, source metrics.txt, chart color)
MODELS = [
    ("HuBERT SER", os.path.join(RESULTS_DIR, "HubertSER_RAVDESS", "metrics.txt"), "#0F766E"),
    ("Yassmen (CREMA-D)", os.path.join(RESULTS_DIR, "Yassmen_RAVDESS", "metrics.txt"), "#7C3AED"),
    ("audEERING (dimensional)", os.path.join(RESULTS_DIR, "Audeering_RAVDESS", "metrics.txt"), "#059669"),
    ("GoEmotions (text baseline)", os.path.join(RESULTS_DIR, "GoEmotions_RAVDESS", "metrics.txt"), "#DB2777"),
]

URBANPULSE_STATES = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]

SUMMARY_FIELD_PATTERNS = {
    "accuracy": r"^Accuracy\s*:\s*([\d.]+)",
    "precision": r"^Precision\s*:\s*([\d.]+)",
    "recall": r"^Recall\s*:\s*([\d.]+)",
    "f1": r"^F1\s*:\s*([\d.]+)",
    "avg_time_ms": r"^Avg Time\s*:\s*([\d.]+)\s*ms",
    "n_clips": r"^Clips evaluated:\s*(\d+)",
}


# Reads back the metrics.txt written by a single-model run.
def parse_metrics_file(path):
    with open(path) as f:
        text = f.read()

    summary = {}
    for key, pattern in SUMMARY_FIELD_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            raise ValueError(f"Could not find '{key}' in {path}")
        val = m.group(1)
        summary[key] = int(val) if key == "n_clips" else float(val)

    per_class = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("UrbanPulse State") and "Precision" in line:
            in_table = True
            continue
        if in_table and line.startswith("-") and per_class:
            break
        if not in_table or not line.strip() or line.startswith("-"):
            continue

        clean_line = line.split("[")[0]
        parts = clean_line.split()
        # parts = [..state name words.., precision, recall, f1, support]
        support = int(parts[-1])
        f1 = float(parts[-2])
        recall = float(parts[-3])
        precision = float(parts[-4])
        state = " ".join(parts[:-4])
        per_class[state] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }

    return summary, per_class


# Merges every model's metrics into one comparison table.
def build_combined_report(all_summaries, all_per_class):
    lines = []
    lines.append("=" * 100)
    lines.append("Model Comparison — RAVDESS — Step 3: Emotion Detection from Speech")
    lines.append("Models: " + " | ".join(name for name, _, _ in MODELS))
    lines.append("All 4 evaluated on the same 500-clip seed=42 sample, none tested on its own training data")
    lines.append("=" * 100)
    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Model':<28}{'Accuracy':>12}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Avg Time (ms)':>16}")
    lines.append("-" * 100)
    for name, _, _ in MODELS:
        s = all_summaries[name]
        lines.append(
            f"{name:<28}{s['accuracy']:>12.3f}{s['precision']:>12.3f}{s['recall']:>10.3f}"
            f"{s['f1']:>10.3f}{s['avg_time_ms']:>16.0f}"
        )
    lines.append("-" * 100)
    ranked = sorted(MODELS, key=lambda m: all_summaries[m[0]]["f1"], reverse=True)
    lines.append(f"Ranked by F1: " + " > ".join(f"{name} ({all_summaries[name]['f1']:.3f})" for name, _, _ in ranked))
    lines.append("-" * 100)
    lines.append("")
    lines.append("PER-CLASS BREAKDOWN (UrbanPulse wellbeing states)")
    lines.append("-" * 100)
    for state in URBANPULSE_STATES:
        first = all_per_class[MODELS[0][0]].get(state)
        support = first["support"] if first else "N/A"
        lines.append(f"\n{state}  (Support: {support})")
        for name, _, _ in MODELS:
            c = all_per_class[name].get(state)
            if c is None:
                lines.append(f"  {name:<26} (no data)")
                continue
            lines.append(f"  {name:<26} P={c['precision']:.3f}  R={c['recall']:.3f}  F1={c['f1']:.3f}")
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


# Charts the headline metric per model, side by side.
def plot_overall_comparison(all_summaries):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 1]})

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    n = len(MODELS)
    width = 0.8 / n

    ax = axes[0]
    for i, (name, _, color) in enumerate(MODELS):
        s = all_summaries[name]
        values = [s[m] for m in metrics]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy / Precision / Recall / F1")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = [name for name, _, _ in MODELS]
    times = [all_summaries[name]["avg_time_ms"] for name in names]
    colors = [color for _, _, color in MODELS]
    bars = ax2.bar(names, times, color=colors)
    ax2.set_ylabel("ms per clip")
    ax2.set_title("Avg Inference Time")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    for bar, val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.0f}", ha="center", fontsize=8)

    fig.suptitle("RAVDESS — Overall Model Comparison (fair, unseen-corpus test, 500 clips)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "overall_summary_comparison.png"), dpi=150)
    plt.close()


# Charts each model's per-class scores side by side.
def plot_per_class_comparison(all_per_class, metric_key, metric_label, filename):
    x = np.arange(len(URBANPULSE_STATES))
    n = len(MODELS)
    width = 0.8 / n

    plt.figure(figsize=(11, 6))
    for i, (name, _, color) in enumerate(MODELS):
        values = []
        for state in URBANPULSE_STATES:
            c = all_per_class[name].get(state)
            values.append(c[metric_key] if c else 0)
        offset = (i - (n - 1) / 2) * width
        plt.bar(x + offset, values, width, label=name, color=color)

    plt.xticks(x, URBANPULSE_STATES)
    plt.ylim(0, 1.05)
    plt.ylabel(metric_label)
    plt.title(f"RAVDESS — Per-Class {metric_label} Comparison (fair, unseen-corpus test)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, filename), dpi=150)
    plt.close()


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    all_summaries, all_per_class = {}, {}
    for name, path, _ in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metrics.txt for {name}: {path}. Run its test_*_ravdess.py script first.")
        summary, per_class = parse_metrics_file(path)
        all_summaries[name] = summary
        all_per_class[name] = per_class
        print(f"Parsed {name}: {len(per_class)} states")

    report_text = build_combined_report(all_summaries, all_per_class)
    print("\n" + report_text)

    with open(os.path.join(COMPARISON_DIR, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    plot_overall_comparison(all_summaries)
    plot_per_class_comparison(all_per_class, "precision", "Precision", "per_class_precision_comparison.png")
    plot_per_class_comparison(all_per_class, "recall", "Recall", "per_class_recall_comparison.png")
    plot_per_class_comparison(all_per_class, "f1", "F1", "per_class_f1_comparison.png")

    print(f"\nCombined comparison saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
