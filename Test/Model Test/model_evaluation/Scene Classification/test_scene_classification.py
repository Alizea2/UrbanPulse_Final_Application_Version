"""Step 1b benchmark: runs a candidate zero-shot model over the scene eval set.
Evaluation only -- not part of the app.
"""

import argparse
import csv
import os
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Scene Classification
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
PROJECT_ROOT = os.path.dirname(TESTS_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from models.scene_classifier import CANDIDATE_SCENES

DATASET_ROOT = os.path.join(TESTS_ROOT, "Datasets", "Scene Classification")
DATASET_CSV = os.path.join(DATASET_ROOT, "scene_eval_set.csv")

# display name (used for results-dir naming) -> HF model id
MODEL_DISPLAY_NAMES = {
    "openai/clip-vit-base-patch32": "CLIP-ViT-B32",
    "openai/clip-vit-large-patch14": "CLIP-ViT-L14",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": "OpenCLIP-ViT-B32-LAION2B",
    "google/siglip-base-patch16-224": "SigLIP-B16",
}


# Loads the evaluation clips and their labels.
def load_dataset(sample=None):
    with open(DATASET_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if sample:
        import random
        random.seed(42)
        rows = random.sample(rows, min(sample, len(rows)))
    return rows


# Runs the model over every clip and collects predictions and timings.
def run_evaluation(clf, rows):
    from PIL import Image

    results = []
    t_start = time.time()

    for i, row in enumerate(rows, start=1):
        image_path = os.path.join(DATASET_ROOT, row["image_path"])
        image = Image.open(image_path).convert("RGB")
        predictions = clf(image, candidate_labels=CANDIDATE_SCENES)
        top1 = predictions[0]
        correct = top1["label"] == row["expected_scene_label"]

        results.append({
            "id": row["id"],
            "expected": row["expected_scene_label"],
            "places365_category": row["places365_category"],
            "predicted": top1["label"],
            "confidence": top1["score"],
            "correct": correct,
        })

        if i % 50 == 0 or i == len(rows):
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(rows) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(rows)} rows | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    return results


# Computes the headline metrics and writes metrics.txt.
def report_metrics(model_name, results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0

    per_scene = {}
    for scene in CANDIDATE_SCENES:
        rows = [r for r in results if r["expected"] == scene]
        if not rows:
            continue
        hit = sum(1 for r in rows if r["correct"])
        per_scene[scene] = (hit, len(rows), hit / len(rows))

    # confusion: for each expected scene, what's the most common WRONG prediction
    confusions = defaultdict(lambda: defaultdict(int))
    for r in results:
        if not r["correct"]:
            confusions[r["expected"]][r["predicted"]] += 1

    lines = []
    lines.append("=" * 90)
    lines.append(f"Scene-Classification Evaluation — {model_name} — Step 1b: Scene Classification")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Grades each of the 420 scene_eval_set.csv rows (real Places365 photos): does the")
    lines.append("model's top-1 prediction out of all 21 CANDIDATE_SCENES match the expected label?")
    lines.append("")
    lines.append("-" * 90)
    lines.append("SUMMARY")
    lines.append("-" * 90)
    lines.append(f"Top-1 accuracy : {accuracy:.4f}  ({correct}/{total} rows)")
    lines.append("-" * 90)
    lines.append("")
    lines.append("PER-SCENE BREAKDOWN")
    lines.append("-" * 90)
    lines.append(f"{'Scene':<32}{'Correct':>10}{'Total':>8}{'Accuracy':>12}")
    lines.append("-" * 90)
    for scene in CANDIDATE_SCENES:
        if scene in per_scene:
            hit, tot, acc = per_scene[scene]
            lines.append(f"{scene:<32}{hit:>10d}{tot:>8d}{acc:>12.3f}")
    lines.append("-" * 90)
    lines.append("")
    lines.append("WORST 10 SCENES (lowest accuracy) AND THEIR TOP CONFUSION")
    lines.append("-" * 90)
    worst_scenes = sorted(per_scene.items(), key=lambda kv: kv[1][2])[:10]
    for scene, (hit, tot, acc) in worst_scenes:
        top_confusion = max(confusions[scene].items(), key=lambda kv: kv[1]) if confusions[scene] else None
        confusion_str = f"most often confused with '{top_confusion[0]}' ({top_confusion[1]}x)" if top_confusion else "no consistent confusion"
        lines.append(f"  {scene:<32} {hit}/{tot} = {acc:.3f}  —  {confusion_str}")
    lines.append("")
    lines.append("=" * 90)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    return report_text, {"accuracy": accuracy, "correct": correct, "total": total, "per_scene": per_scene}


# Writes the result charts used in the report.
def save_graphs(model_name, results, metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    scenes = list(metrics["per_scene"].keys())
    accs = [metrics["per_scene"][s][2] for s in scenes]
    order = np.argsort(accs)
    scenes_sorted = [scenes[i] for i in order]
    accs_sorted = [accs[i] for i in order]
    colors = ["#DC2626" if a < 0.5 else "#059669" for a in accs_sorted]
    plt.barh(scenes_sorted, accs_sorted, color=colors)
    plt.xlabel("Top-1 accuracy")
    plt.xlim(0, 1.05)
    plt.title(f"{model_name} — Per-Scene Top-1 Accuracy (overall={metrics['accuracy']*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_scene_accuracy.png"), dpi=150)
    plt.close()

    print(f"\nGraphs and metrics.txt saved to: {results_dir}")


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Scene-classification evaluation for Step 1b")
    parser.add_argument("--model", required=True, choices=list(MODEL_DISPLAY_NAMES.keys()),
                         help="HF model id to test")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (default: all 420 rows)")
    args = parser.parse_args()

    display_name = MODEL_DISPLAY_NAMES[args.model]
    print(f"Testing model: {args.model} ({display_name})")

    print("Loading pipeline...")
    from transformers import pipeline
    clf = pipeline("zero-shot-image-classification", model=args.model, device=-1)

    results_dir = os.path.join(TESTS_ROOT, "Test-Results", "Scene Classification", f"{display_name}_Places365")

    rows = load_dataset(args.sample)
    print(f"Evaluating {len(rows)} rows from scene_eval_set.csv...")
    results = run_evaluation(clf, rows)

    report_text, metrics = report_metrics(display_name, results)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    save_graphs(display_name, results, metrics, results_dir)


if __name__ == "__main__":
    main()
