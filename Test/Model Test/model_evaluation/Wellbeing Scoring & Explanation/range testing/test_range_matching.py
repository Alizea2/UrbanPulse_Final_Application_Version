"""Range-matching evaluation for the Step 4 wellbeing scorer: scores each row
of the curated eval set and checks it falls inside the expected range.
Evaluation only -- not part of the app.
"""

import argparse
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Wellbeing Scoring & Explanation/range testing
TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
PROJECT_ROOT = os.path.dirname(TESTS_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import models.wellbeing_scorer as wellbeing_scorer
from models.wellbeing_scorer import (
    PLACE_PROMPT_WITH_SCENE, PERSON_PROMPT_TEMPLATE,
    SOUND_SCORE_RUBRIC, EMOTION_SCORE_RUBRIC, SCENE_SCORE_RUBRIC, TRANSCRIPT_SCORE_RUBRIC,
    _call_ollama, _extract_json, combine_scores,
)

DATASET_PATH = os.path.join(TESTS_ROOT, "Datasets", "Wellbeing Scoring & Explanation", "wellbeing_eval_set_4input.csv")

URBANPULSE_STATES = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]


# Loads the evaluation clips and their labels.
def load_dataset(sample=None):
    with open(DATASET_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if sample:
        import random
        random.seed(42)
        rows = random.sample(rows, min(sample, len(rows)))
    return rows


def get_wellbeing_score(sound_label, emotion, transcript, scene_label):
    """Runs only the two sub-score calls (Place: sound+scene, Person:
    emotion+transcript) + the deterministic combine — skips the
    explanation call, irrelevant to range-matching. Same helper pattern
    as the consistency suite, and mirrors
    models.wellbeing_scorer.score_wellbeing()'s first two stages
    exactly."""
    place_prompt = PLACE_PROMPT_WITH_SCENE.format(
        sound_label=sound_label, scene_label=scene_label,
        sound_rubric=SOUND_SCORE_RUBRIC, scene_rubric=SCENE_SCORE_RUBRIC,
    )
    raw_place = _call_ollama(place_prompt, temperature=0.0)
    parsed_place = _extract_json(raw_place)
    sound_score = max(0.0, min(1.0, float(parsed_place["sound_score"])))
    scene_score = max(0.0, min(1.0, float(parsed_place["scene_score"])))

    person_prompt = PERSON_PROMPT_TEMPLATE.format(
        emotion=emotion, transcript=transcript,
        emotion_rubric=EMOTION_SCORE_RUBRIC, transcript_rubric=TRANSCRIPT_SCORE_RUBRIC,
    )
    raw_person = _call_ollama(person_prompt, temperature=0.0)
    parsed_person = _extract_json(raw_person)
    emotion_score = max(0.0, min(1.0, float(parsed_person["emotion_score"])))
    transcript_score = max(0.0, min(1.0, float(parsed_person["transcript_score"])))

    return combine_scores(sound_score, emotion_score, transcript_score, scene_score)


# Runs the model over every clip and collects predictions and timings.
def run_evaluation(rows):
    results = []
    t_start = time.time()

    for i, row in enumerate(rows, start=1):
        score = get_wellbeing_score(row["sound_label"], row["emotion"], row["transcript"], row["scene_label"])
        range_min = float(row["acceptable_score_min"])
        range_max = float(row["acceptable_score_max"])
        in_range = range_min <= score <= range_max

        results.append({
            "id": row["id"],
            "sound_label": row["sound_label"],
            "emotion": row["emotion"],
            "scene_label": row["scene_label"],
            "predicted_score": score,
            "range_min": range_min,
            "range_max": range_max,
            "in_range": in_range,
            "miss_distance": 0.0 if in_range else min(abs(score - range_min), abs(score - range_max)),
        })

        if i % 25 == 0 or i == len(rows):
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(rows) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(rows)} rows | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    return results


# Computes the headline metrics and writes metrics.txt.
def report_metrics(model_name, results):
    total = len(results)
    correct = sum(1 for r in results if r["in_range"])
    accuracy = correct / total if total else 0.0

    misses = [r["miss_distance"] for r in results if not r["in_range"]]
    mean_miss = sum(misses) / len(misses) if misses else 0.0

    per_emotion = {}
    for state in URBANPULSE_STATES:
        rows = [r for r in results if r["emotion"] == state]
        if not rows:
            continue
        hit = sum(1 for r in rows if r["in_range"])
        per_emotion[state] = (hit, len(rows), hit / len(rows))

    lines = []
    lines.append("=" * 90)
    lines.append(f"Range-Matching Evaluation — {model_name} — Step 4: Wellbeing Scoring & Explanation")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Grades each of the 115 wellbeing_eval_set_4input.csv rows: is the model's predicted score")
    lines.append("within that row's [acceptable_score_min, acceptable_score_max] range?")
    lines.append("")
    lines.append("-" * 90)
    lines.append("SUMMARY")
    lines.append("-" * 90)
    lines.append(f"In-range accuracy : {accuracy:.4f}  ({correct}/{total} rows)")
    lines.append(f"Mean miss distance (rows outside range only): {mean_miss:.4f}")
    lines.append("-" * 90)
    lines.append("")
    lines.append("PER-EMOTION BREAKDOWN")
    lines.append("-" * 90)
    lines.append(f"{'Emotion':<12}{'In-range':>12}{'Total':>10}{'Accuracy':>12}")
    lines.append("-" * 90)
    for state in URBANPULSE_STATES:
        if state in per_emotion:
            hit, tot, acc = per_emotion[state]
            lines.append(f"{state:<12}{hit:>12d}{tot:>10d}{acc:>12.3f}")
    lines.append("-" * 90)
    lines.append("")
    lines.append("WORST 10 MISSES (largest miss distance)")
    lines.append("-" * 90)
    worst = sorted([r for r in results if not r["in_range"]], key=lambda r: -r["miss_distance"])[:10]
    for r in worst:
        lines.append(f"  #{r['id']:<4} {r['sound_label']:<28}{r['emotion']:<10}{r['scene_label']:<24} "
                      f"predicted={r['predicted_score']:.2f}  range=[{r['range_min']:.2f}, {r['range_max']:.2f}]  "
                      f"miss={r['miss_distance']:.2f}")
    lines.append("")
    lines.append("=" * 90)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    return report_text, {"accuracy": accuracy, "correct": correct, "total": total, "mean_miss": mean_miss, "per_emotion": per_emotion}


# Writes the result charts used in the report.
def save_graphs(model_name, results, metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # Predicted score vs. acceptable range, per row (sorted by row id)
    plt.figure(figsize=(16, 6))
    ids = [int(r["id"]) for r in results]
    predicted = [r["predicted_score"] for r in results]
    mins = [r["range_min"] for r in results]
    maxs = [r["range_max"] for r in results]
    colors = ["#059669" if r["in_range"] else "#DC2626" for r in results]

    plt.vlines(ids, mins, maxs, color="#94a3b8", alpha=0.5, linewidth=2, label="Acceptable range")
    plt.scatter(ids, predicted, c=colors, s=14, zorder=3, label="Predicted score")
    plt.xlabel("Dataset row id")
    plt.ylabel("Wellbeing Score")
    plt.ylim(-0.05, 1.05)
    plt.title(f"{model_name} — Predicted Score vs. Acceptable Range per Row\n"
              f"(green = in range, red = miss) — accuracy={metrics['accuracy']*100:.1f}%")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_vs_range.png"), dpi=150)
    plt.close()

    # Per-emotion accuracy bar chart
    plt.figure(figsize=(8, 6))
    states = list(metrics["per_emotion"].keys())
    accs = [metrics["per_emotion"][s][2] for s in states]
    plt.bar(states, accs, color="#7C3AED")
    plt.ylim(0, 1.05)
    plt.ylabel("In-range accuracy")
    plt.title(f"{model_name} — Range-Matching Accuracy by Emotion")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_emotion_accuracy.png"), dpi=150)
    plt.close()

    print(f"\nGraphs and metrics.txt saved to: {results_dir}")


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Range-matching evaluation for Step 4")
    parser.add_argument("--model", default=wellbeing_scorer.MODEL_NAME,
                         help="Ollama model tag to test (must already be pulled)")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (default: all 115 rows)")
    args = parser.parse_args()

    wellbeing_scorer.MODEL_NAME = args.model
    print(f"Testing model: {args.model}")

    results_dir = os.path.join(TESTS_ROOT, "Test-Results", "Wellbeing Scoring & Explanation", "range testing",
                                f"{args.model.replace(':', '_')}_Range")

    rows = load_dataset(args.sample)
    print(f"Evaluating {len(rows)} rows from wellbeing_eval_set_4input.csv...")
    results = run_evaluation(rows)

    report_text, metrics = report_metrics(args.model, results)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    save_graphs(args.model, results, metrics, results_dir)


if __name__ == "__main__":
    main()
