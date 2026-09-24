"""Merges the LibriSpeech test-clean transcription evaluations into one
metrics.txt and one set of comparison graphs.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Voice Note Transcription")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "Model_Comparison_LibriSpeech")

# (display name, source metrics.txt, chart color)
MODELS = [
    ("wav2vec2-base", os.path.join(RESULTS_DIR, "wav2vec2_LibriSpeech", "metrics.txt"), "#0F766E"),
    ("Distil-Whisper", os.path.join(RESULTS_DIR, "DistilWhisper_LibriSpeech", "metrics.txt"), "#7C3AED"),
    ("Whisper small", os.path.join(RESULTS_DIR, "Whisper_small_LibriSpeech", "metrics.txt"), "#059669"),
    ("wav2vec2-large", os.path.join(RESULTS_DIR, "wav2vec2large_LibriSpeech", "metrics.txt"), "#B45309"),
]

FIELD_PATTERNS = {
    "wer": r"^WER \(Word Error Rate\)\s*:\s*([\d.]+)",
    "cer": r"^CER \(Character Error Rate\):\s*([\d.]+)",
    "perfect_n": r"^Perfect transcriptions\s*:\s*(\d+)/(\d+)",
    "avg_time_ms": r"^Avg Time\s*:\s*([\d.]+)\s*ms",
}


# Reads back the metrics.txt written by a single-model run.
def parse_metrics_file(path):
    with open(path) as f:
        text = f.read()

    result = {}
    for key, pattern in FIELD_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            raise ValueError(f"Could not find '{key}' in {path}")
        if key == "perfect_n":
            perfect, total = int(m.group(1)), int(m.group(2))
            result["perfect_rate"] = perfect / total
            result["n_clips"] = total
        else:
            result[key] = float(m.group(1))
    return result


# Merges every model's metrics into one comparison table.
def build_combined_report(all_metrics):
    ranked = sorted(MODELS, key=lambda m: all_metrics[m[0]]["wer"])

    lines = []
    lines.append("=" * 90)
    lines.append("Model Comparison — LibriSpeech test-clean — Step 2: Voice Note Transcription")
    lines.append("Models: " + " | ".join(name for name, _, _ in MODELS))
    lines.append("=" * 90)
    lines.append("")
    lines.append("OVERALL SUMMARY (ranked best to worst by WER)")
    lines.append("-" * 90)
    lines.append(f"{'Rank':<6}{'Model':<18}{'WER':>10}{'CER':>10}{'Perfect Match':>16}{'Avg Time (ms)':>16}")
    lines.append("-" * 90)
    for i, (name, _, _) in enumerate(ranked, start=1):
        m = all_metrics[name]
        lines.append(
            f"{i:<6}{name:<18}{m['wer']*100:>9.2f}%{m['cer']*100:>9.2f}%"
            f"{m['perfect_rate']*100:>15.1f}%{m['avg_time_ms']:>16.0f}"
        )
    lines.append("-" * 90)
    lines.append(f"\nAll models evaluated on the same {all_metrics[MODELS[0][0]]['n_clips']}-clip random sample "
                  f"(seed=42) of LibriSpeech test-clean, using Whisper's official EnglishTextNormalizer "
                  f"before scoring, for direct comparability.")
    lines.append("")
    lines.append("=" * 90)

    return "\n".join(lines)


# Charts the headline metric per model, side by side.
def plot_overall_comparison(all_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    metrics = ["wer", "cer", "perfect_rate"]
    metric_labels = ["WER", "CER", "Perfect Match Rate"]
    x = np.arange(len(metrics))
    n = len(MODELS)
    width = 0.8 / n

    ax = axes[0]
    for i, (name, _, color) in enumerate(MODELS):
        m = all_metrics[name]
        values = [m[k] for k in metrics]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    max_val = max(all_metrics[model_name][k] for model_name, _, _ in MODELS for k in metrics)
    ax.set_ylim(0, max_val * 1.3)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("WER / CER / Perfect Match Rate")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = [name for name, _, _ in MODELS]
    times = [all_metrics[name]["avg_time_ms"] for name in names]
    colors = [color for _, _, color in MODELS]
    bars = ax2.bar(names, times, color=colors)
    ax2.set_ylabel("ms per clip")
    ax2.set_title("Avg Inference Time")
    ax2.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 20, f"{val:.0f}", ha="center", fontsize=8)

    fig.suptitle("LibriSpeech test-clean — Overall Model Comparison (500-clip sample)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "overall_summary_comparison.png"), dpi=150)
    plt.close()


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    all_metrics = {}
    for name, path, _ in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metrics.txt for {name}: {path}. Run its test_*_librispeech.py script first.")
        all_metrics[name] = parse_metrics_file(path)
        print(f"Parsed {name}: WER={all_metrics[name]['wer']:.4f}")

    report_text = build_combined_report(all_metrics)
    print("\n" + report_text)

    with open(os.path.join(COMPARISON_DIR, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    plot_overall_comparison(all_metrics)

    print(f"\nCombined comparison saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
