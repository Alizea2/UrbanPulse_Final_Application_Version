"""Merges the four SONYC-UST evaluations (YAMNet, PANNs CNN14, E-PANNs,
EfficientAT) into one metrics.txt and one set of comparison graphs.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Tests/model_evaluation/Sound Classification
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../Tests
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Sound Classification")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "Model_Comparison_SONYC-UST")

# (display name, source metrics.txt, chart color)
MODELS = [
    ("YAMNet", os.path.join(RESULTS_DIR, "YAMNet_SONYC-UST", "metrics.txt"), "#0F766E"),
    ("PANNs CNN14", os.path.join(RESULTS_DIR, "PANNs_SONYC-UST", "metrics.txt"), "#7C3AED"),
    ("E-PANNs", os.path.join(RESULTS_DIR, "EPANNs_SONYC-UST", "metrics.txt"), "#059669"),
    ("EfficientAT", os.path.join(RESULTS_DIR, "EfficientAT_SONYC-UST", "metrics.txt"), "#DB2777"),
]

SUMMARY_FIELD_PATTERNS = {
    "accuracy": r"^Accuracy\s*:\s*([\d.]+)",
    "precision": r"^Precision\s*:\s*([\d.]+)",
    "recall": r"^Recall\s*:\s*([\d.]+)",
    "f1": r"^F1\s*:\s*([\d.]+)",
    "auc": r"^AUC\s*:\s*([\d.]+|N/A)",
    "avg_time_ms": r"^Avg Time\s*:\s*([\d.]+)\s*ms",
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
        summary[key] = None if val == "N/A" else float(val)

    per_class = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("Class") and "Precision" in line:
            in_table = True
            continue
        if in_table and line.startswith("="):
            break
        if not in_table or not line.strip() or line.startswith("-"):
            continue

        no_mapping = "[no mapping]" in line
        clean = line.replace("[no mapping]", "").split()
        # clean = [class_name, precision, recall, f1, auc, support]
        cls, p, r, f1, auc, support = clean
        per_class[cls] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "auc": None if auc == "N/A" else float(auc),
            "support": int(support),
            "no_mapping": no_mapping,
        }

    return summary, per_class


# Merges every model's metrics into one comparison table.
def build_combined_report(all_summaries, all_per_class, class_order):
    lines = []
    lines.append("=" * 100)
    lines.append("Model Comparison — SONYC-UST — Step 1: Sound Classification (23 fine-grained classes)")
    lines.append("Models: " + " | ".join(name for name, _, _ in MODELS))
    lines.append("=" * 100)
    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Model':<16}{'Accuracy':>12}{'Precision':>12}{'Recall':>10}{'F1':>10}{'AUC':>10}{'Avg Time (ms)':>16}")
    lines.append("-" * 100)
    for name, _, _ in MODELS:
        s = all_summaries[name]
        auc_str = f"{s['auc']:.3f}" if s["auc"] is not None else "N/A"
        lines.append(
            f"{name:<16}{s['accuracy']:>12.3f}{s['precision']:>12.3f}{s['recall']:>10.3f}"
            f"{s['f1']:>10.3f}{auc_str:>10}{s['avg_time_ms']:>16.0f}"
        )
    lines.append("-" * 100)
    lines.append("")
    lines.append("PER-CLASS BREAKDOWN")
    lines.append("-" * 100)
    for cls in class_order:
        first = all_per_class[MODELS[0][0]][cls]
        flag = "  [no mapping]" if first["no_mapping"] else ""
        lines.append(f"\n{cls}  (Support: {first['support']}){flag}")
        for name, _, _ in MODELS:
            c = all_per_class[name][cls]
            auc_str = f"{c['auc']:.3f}" if c["auc"] is not None else "N/A"
            lines.append(f"  {name:<14} P={c['precision']:.3f}  R={c['recall']:.3f}  F1={c['f1']:.3f}  AUC={auc_str}")
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


# Charts the headline metric per model, side by side.
def plot_overall_comparison(all_summaries):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    x = np.arange(len(metrics))
    n = len(MODELS)
    width = 0.8 / n

    ax = axes[0]
    for i, (name, _, color) in enumerate(MODELS):
        s = all_summaries[name]
        values = [s[m] if s[m] is not None else 0 for m in metrics]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy / Precision / Recall / F1 / AUC")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = [name for name, _, _ in MODELS]
    times = [all_summaries[name]["avg_time_ms"] for name in names]
    colors = [color for _, _, color in MODELS]
    bars = ax2.bar(names, times, color=colors)
    ax2.set_ylabel("ms per clip")
    ax2.set_title("Avg Inference Time")
    ax2.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.0f}", ha="center", fontsize=8)

    fig.suptitle("SONYC-UST — Overall Model Comparison (top-3, test split, 664 clips)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "overall_summary_comparison.png"), dpi=150)
    plt.close()


# Charts each model's per-class scores side by side.
def plot_per_class_comparison(all_per_class, class_order, metric_key, metric_label, filename):
    x = np.arange(len(class_order))
    n = len(MODELS)
    width = 0.8 / n

    plt.figure(figsize=(20, 8))
    for i, (name, _, color) in enumerate(MODELS):
        values = []
        for cls in class_order:
            v = all_per_class[name][cls][metric_key]
            values.append(v if v is not None else 0)
        offset = (i - (n - 1) / 2) * width
        plt.bar(x + offset, values, width, label=name, color=color)

    labels = [f"{c}*" if all_per_class[MODELS[0][0]][c]["no_mapping"] else c for c in class_order]
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel(metric_label)
    plt.title(f"SONYC-UST — Per-Class {metric_label} Comparison (top-3, test split)\n(* = no possible mapping for any model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, filename), dpi=150)
    plt.close()


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    all_summaries, all_per_class = {}, {}
    for name, path, _ in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metrics.txt for {name}: {path}. Run its test_*_sonyc_ust.py script first.")
        summary, per_class = parse_metrics_file(path)
        all_summaries[name] = summary
        all_per_class[name] = per_class
        print(f"Parsed {name}: {len(per_class)} classes")

    class_order = list(all_per_class[MODELS[0][0]].keys())

    report_text = build_combined_report(all_summaries, all_per_class, class_order)
    print("\n" + report_text)

    with open(os.path.join(COMPARISON_DIR, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    plot_overall_comparison(all_summaries)
    plot_per_class_comparison(all_per_class, class_order, "precision", "Precision", "per_class_precision_comparison.png")
    plot_per_class_comparison(all_per_class, class_order, "recall", "Recall", "per_class_recall_comparison.png")
    plot_per_class_comparison(all_per_class, class_order, "f1", "F1", "per_class_f1_comparison.png")
    plot_per_class_comparison(all_per_class, class_order, "auc", "AUC", "per_class_auc_comparison.png")

    print(f"\nCombined comparison saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
