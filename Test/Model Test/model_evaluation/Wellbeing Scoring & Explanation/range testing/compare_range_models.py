"""Merges the Step 4 range-matching evaluations (Phi-3-mini, gemma2:2b,
qwen2.5:3b, llama3.2:3b) into one metrics.txt and one set of graphs.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Wellbeing Scoring & Explanation/range testing
TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Wellbeing Scoring & Explanation", "range testing")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "Model_Comparison_Range")

MODELS = [
    ("Phi-3-mini", os.path.join(RESULTS_DIR, "phi3_mini_Range", "metrics.txt"), "#0F766E"),
    ("gemma2:2b", os.path.join(RESULTS_DIR, "gemma2_2b_Range", "metrics.txt"), "#7C3AED"),
    ("qwen2.5:3b", os.path.join(RESULTS_DIR, "qwen2.5_3b_Range", "metrics.txt"), "#059669"),
    ("llama3.2:3b", os.path.join(RESULTS_DIR, "llama3.2_3b_Range", "metrics.txt"), "#DB2777"),
]

URBANPULSE_STATES = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]

SUMMARY_PATTERNS = {
    "accuracy": r"^In-range accuracy\s*:\s*([\d.]+)",
    "mean_miss": r"^Mean miss distance.*:\s*([\d.]+)",
}


# Reads back the metrics.txt written by a single-model run.
def parse_metrics_file(path):
    with open(path) as f:
        text = f.read()

    summary = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            raise ValueError(f"Could not find '{key}' in {path}")
        summary[key] = float(m.group(1))

    per_emotion = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("Emotion") and "In-range" in line:
            in_table = True
            continue
        if in_table and line.startswith("-") and per_emotion:
            break
        if not in_table or not line.strip() or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        emotion, hit, total, acc = parts
        per_emotion[emotion] = {"hit": int(hit), "total": int(total), "accuracy": float(acc)}

    return summary, per_emotion


# Merges every model's metrics into one comparison table.
def build_combined_report(all_summaries, all_per_emotion):
    lines = []
    lines.append("=" * 100)
    lines.append("Model Comparison — Range-Matching — Step 4: Wellbeing Scoring & Explanation")
    lines.append("Models: " + " | ".join(name for name, _, _ in MODELS))
    lines.append("=" * 100)
    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Model':<16}{'In-range Accuracy':>20}{'Mean Miss Distance':>22}")
    lines.append("-" * 100)
    for name, _, _ in MODELS:
        s = all_summaries[name]
        lines.append(f"{name:<16}{s['accuracy']:>20.4f}{s['mean_miss']:>22.4f}")
    lines.append("-" * 100)
    ranked = sorted(MODELS, key=lambda m: all_summaries[m[0]]["accuracy"], reverse=True)
    lines.append("Ranked by in-range accuracy: " + " > ".join(
        f"{name} ({all_summaries[name]['accuracy']*100:.1f}%)" for name, _, _ in ranked))
    lines.append("-" * 100)
    lines.append("")
    lines.append("PER-EMOTION BREAKDOWN")
    lines.append("-" * 100)
    for state in URBANPULSE_STATES:
        lines.append(f"\n{state}")
        for name, _, _ in MODELS:
            e = all_per_emotion[name].get(state)
            if e is None:
                lines.append(f"  {name:<16} (no data)")
                continue
            lines.append(f"  {name:<16} {e['hit']}/{e['total']} = {e['accuracy']:.3f}")
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


# Charts the headline metric per model, side by side.
def plot_overall_comparison(all_summaries):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    names = [name for name, _, _ in MODELS]
    accs = [all_summaries[name]["accuracy"] for name in names]
    colors = [color for _, _, color in MODELS]
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("In-range accuracy")
    ax.set_title("Range-Matching Accuracy (115 rows)")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val*100:.1f}%", ha="center", fontsize=9)

    ax2 = axes[1]
    misses = [all_summaries[name]["mean_miss"] for name in names]
    bars2 = ax2.bar(names, misses, color=colors)
    ax2.set_ylabel("Mean miss distance (rows outside range)")
    ax2.set_title("Severity of Misses")
    ax2.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars2, misses):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle("Step 4 — Range-Matching: Model Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "overall_comparison.png"), dpi=150)
    plt.close()


# Charts each model's per-emotion scores side by side.
def plot_per_emotion_comparison(all_per_emotion):
    x = np.arange(len(URBANPULSE_STATES))
    n = len(MODELS)
    width = 0.8 / n

    plt.figure(figsize=(11, 6))
    for i, (name, _, color) in enumerate(MODELS):
        values = []
        for state in URBANPULSE_STATES:
            e = all_per_emotion[name].get(state)
            values.append(e["accuracy"] if e else 0)
        offset = (i - (n - 1) / 2) * width
        plt.bar(x + offset, values, width, label=name, color=color)

    plt.xticks(x, URBANPULSE_STATES)
    plt.ylim(0, 1.05)
    plt.ylabel("In-range accuracy")
    plt.title("Range-Matching Accuracy by Emotion")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "per_emotion_comparison.png"), dpi=150)
    plt.close()


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    all_summaries, all_per_emotion = {}, {}
    for name, path, _ in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metrics.txt for {name}: {path}. Run test_range_matching.py --model first.")
        summary, per_emotion = parse_metrics_file(path)
        all_summaries[name] = summary
        all_per_emotion[name] = per_emotion
        print(f"Parsed {name}")

    report_text = build_combined_report(all_summaries, all_per_emotion)
    print("\n" + report_text)

    with open(os.path.join(COMPARISON_DIR, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    plot_overall_comparison(all_summaries)
    plot_per_emotion_comparison(all_per_emotion)

    print(f"\nCombined comparison saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
