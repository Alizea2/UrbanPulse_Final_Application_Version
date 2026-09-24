"""Merges the Step 1b scene evaluations (CLIP-ViT-B32, CLIP-ViT-L14,
OpenCLIP-ViT-B32-LAION2B, SigLIP-B16) into one metrics.txt and one set
of comparison graphs.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Scene Classification
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Scene Classification")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "Model_Comparison_Scene")

# (display name, source metrics.txt, chart color)
MODELS = [
    ("CLIP-ViT-B32 (current)", os.path.join(RESULTS_DIR, "CLIP-ViT-B32_Places365", "metrics.txt"), "#059669"),
    ("CLIP-ViT-L14", os.path.join(RESULTS_DIR, "CLIP-ViT-L14_Places365", "metrics.txt"), "#0F766E"),
    ("OpenCLIP-ViT-B32-LAION2B", os.path.join(RESULTS_DIR, "OpenCLIP-ViT-B32-LAION2B_Places365", "metrics.txt"), "#7C3AED"),
    ("SigLIP-B16", os.path.join(RESULTS_DIR, "SigLIP-B16_Places365", "metrics.txt"), "#DB2777"),
]

SUMMARY_PATTERNS = {
    "accuracy": r"^Top-1 accuracy\s*:\s*([\d.]+)",
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

    per_scene = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("Scene") and "Correct" in line:
            in_table = True
            continue
        if in_table and line.startswith("-") and per_scene:
            break
        if not in_table or not line.strip() or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        acc = float(parts[-1])
        total = int(parts[-2])
        hit = int(parts[-3])
        scene = " ".join(parts[:-3])
        per_scene[scene] = {"hit": hit, "total": total, "accuracy": acc}

    return summary, per_scene


# Merges every model's metrics into one comparison table.
def build_combined_report(all_summaries, all_per_scene):
    lines = []
    lines.append("=" * 100)
    lines.append("Model Comparison — Scene Classification — Step 1b: Photo -> Scene Label")
    lines.append("Models: " + " | ".join(name for name, _, _ in MODELS))
    lines.append("=" * 100)
    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Model':<28}{'Top-1 Accuracy':>18}")
    lines.append("-" * 100)
    for name, _, _ in MODELS:
        s = all_summaries[name]
        lines.append(f"{name:<28}{s['accuracy']:>18.4f}")
    lines.append("-" * 100)
    ranked = sorted(MODELS, key=lambda m: all_summaries[m[0]]["accuracy"], reverse=True)
    lines.append("Ranked by top-1 accuracy: " + " > ".join(
        f"{name} ({all_summaries[name]['accuracy']*100:.1f}%)" for name, _, _ in ranked))
    lines.append("-" * 100)
    lines.append("")
    lines.append("PER-SCENE BREAKDOWN")
    lines.append("-" * 100)
    all_scenes = sorted({scene for name, _, _ in MODELS for scene in all_per_scene[name]})
    for scene in all_scenes:
        lines.append(f"\n{scene}")
        for name, _, _ in MODELS:
            e = all_per_scene[name].get(scene)
            if e is None:
                lines.append(f"  {name:<28} (no data)")
                continue
            lines.append(f"  {name:<28} {e['hit']}/{e['total']} = {e['accuracy']:.3f}")
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


# Charts the headline metric per model, side by side.
def plot_overall_comparison(all_summaries):
    plt.figure(figsize=(9, 6))
    names = [name for name, _, _ in MODELS]
    accs = [all_summaries[name]["accuracy"] for name in names]
    colors = [color for _, _, color in MODELS]
    bars = plt.bar(names, accs, color=colors)
    plt.ylim(0, 1.05)
    plt.ylabel("Top-1 accuracy")
    plt.title("Scene-Classification Accuracy (420 rows, Places365)")
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val*100:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "overall_comparison.png"), dpi=150)
    plt.close()


# Charts each model's per-scene scores side by side.
def plot_per_scene_comparison(all_per_scene):
    all_scenes = sorted({scene for name, _, _ in MODELS for scene in all_per_scene[name]})
    x = np.arange(len(all_scenes))
    n = len(MODELS)
    width = 0.8 / n

    plt.figure(figsize=(16, 8))
    for i, (name, _, color) in enumerate(MODELS):
        values = [all_per_scene[name].get(s, {}).get("accuracy", 0) for s in all_scenes]
        offset = (i - (n - 1) / 2) * width
        plt.bar(x + offset, values, width, label=name, color=color)

    plt.xticks(x, all_scenes, rotation=60, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Top-1 accuracy")
    plt.title("Per-Scene Accuracy by Model")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "per_scene_comparison.png"), dpi=150)
    plt.close()


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    all_summaries, all_per_scene = {}, {}
    for name, path, _ in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metrics.txt for {name}: {path}. Run test_scene_classification.py --model first.")
        summary, per_scene = parse_metrics_file(path)
        all_summaries[name] = summary
        all_per_scene[name] = per_scene
        print(f"Parsed {name}")

    report_text = build_combined_report(all_summaries, all_per_scene)
    print("\n" + report_text)

    with open(os.path.join(COMPARISON_DIR, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    plot_overall_comparison(all_summaries)
    plot_per_scene_comparison(all_per_scene)

    print(f"\nCombined comparison saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
