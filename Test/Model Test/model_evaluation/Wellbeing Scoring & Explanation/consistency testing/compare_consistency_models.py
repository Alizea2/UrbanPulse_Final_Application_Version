"""Merges the Step 4 consistency evaluations (Phi-3-mini, gemma2:2b,
qwen2.5:3b) into one metrics.txt and one set of comparison graphs.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Wellbeing Scoring & Explanation/consistency testing
TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Wellbeing Scoring & Explanation", "consistency testing")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "Model_Comparison_Consistency")

# (display name, source metrics.txt, chart color)
MODELS = [
    ("Phi-3-mini", os.path.join(RESULTS_DIR, "phi3_mini_Consistency", "metrics.txt"), "#0F766E"),
    ("gemma2:2b", os.path.join(RESULTS_DIR, "gemma2_2b_Consistency", "metrics.txt"), "#7C3AED"),
    ("qwen2.5:3b", os.path.join(RESULTS_DIR, "qwen2.5_3b_Consistency", "metrics.txt"), "#059669"),
    ("llama3.2:3b", os.path.join(RESULTS_DIR, "llama3.2_3b_Consistency", "metrics.txt"), "#DB2777"),
]

SUMMARY_FIELD_PATTERNS = {
    "sound_rho": r"^Sound-ranking Spearman rho\s*:\s*(-?[\d.]+)",
    "emotion_rho": r"^Emotion-ranking Spearman rho\s*:\s*(-?[\d.]+)",
    "retest_stdev": r"^Test-retest mean stdev\s*:\s*([\d.]+)",
    "extremes_pass_rate": r"^Extremes pass rate\s*:\s*([\d.]+)%",
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
        summary[key] = float(m.group(1))
    return summary


# Merges every model's metrics into one comparison table.
def build_combined_report(all_summaries):
    lines = []
    lines.append("=" * 100)
    lines.append("Model Comparison — Property-Based Consistency — Step 4: Wellbeing Scoring & Explanation")
    lines.append("Models: " + " | ".join(name for name, _, _ in MODELS))
    lines.append("(Mistral 7B and the originally-planned Qwen2.5-7B/Gemma2-9B dropped: too large for this")
    lines.append(" machine's ~7.5GB RAM — Mistral 7B was confirmed to hang/thrash rather than complete.)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Model':<16}{'Sound rho':>14}{'Emotion rho':>14}{'Retest stdev':>16}{'Extremes pass':>16}")
    lines.append("-" * 100)
    for name, _, _ in MODELS:
        s = all_summaries[name]
        lines.append(
            f"{name:<16}{s['sound_rho']:>14.4f}{s['emotion_rho']:>14.4f}"
            f"{s['retest_stdev']:>16.4f}{s['extremes_pass_rate']:>15.1f}%"
        )
    lines.append("-" * 100)
    ranked = sorted(
        MODELS,
        key=lambda m: (all_summaries[m[0]]["extremes_pass_rate"],
                        all_summaries[m[0]]["sound_rho"] + all_summaries[m[0]]["emotion_rho"],
                        -all_summaries[m[0]]["retest_stdev"]),
        reverse=True,
    )
    lines.append("Ranked (by extremes pass rate, then combined rank correlation, then reliability):")
    for i, (name, _, _) in enumerate(ranked, start=1):
        s = all_summaries[name]
        lines.append(f"  {i}. {name} — extremes={s['extremes_pass_rate']:.0f}%, "
                      f"sound_rho={s['sound_rho']:.3f}, emotion_rho={s['emotion_rho']:.3f}, "
                      f"retest_stdev={s['retest_stdev']:.4f}")
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


# Charts the headline metric per model, side by side.
def plot_overall_comparison(all_summaries):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    metrics = ["sound_rho", "emotion_rho", "extremes_pass_rate_frac"]
    metric_labels = ["Sound-ranking\nSpearman rho", "Emotion-ranking\nSpearman rho", "Extremes\npass rate"]
    x = np.arange(len(metrics))
    n = len(MODELS)
    width = 0.8 / n

    ax = axes[0]
    for i, (name, _, color) in enumerate(MODELS):
        s = all_summaries[name]
        values = [s["sound_rho"], s["emotion_rho"], s["extremes_pass_rate"] / 100.0]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Ranking Correlation & Extremes Pass Rate")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = [name for name, _, _ in MODELS]
    stdevs = [all_summaries[name]["retest_stdev"] for name in names]
    colors = [color for _, _, color in MODELS]
    bars = ax2.bar(names, stdevs, color=colors)
    ax2.set_ylabel("Mean stdev across 100 repeat calls")
    ax2.set_title("Test-Retest Reliability\n(lower = more consistent)")
    ax2.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, stdevs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.0005, f"{val:.4f}", ha="center", fontsize=8)

    fig.suptitle("Step 4 — Wellbeing Scoring Consistency: Model Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "overall_comparison.png"), dpi=150)
    plt.close()


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    all_summaries = {}
    for name, path, _ in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metrics.txt for {name}: {path}. Run test_consistency.py --model first.")
        all_summaries[name] = parse_metrics_file(path)
        print(f"Parsed {name}")

    report_text = build_combined_report(all_summaries)
    print("\n" + report_text)

    with open(os.path.join(COMPARISON_DIR, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    plot_overall_comparison(all_summaries)

    print(f"\nCombined comparison saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
