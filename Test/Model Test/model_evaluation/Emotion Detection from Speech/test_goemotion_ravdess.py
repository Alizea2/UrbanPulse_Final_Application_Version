"""Step 3 benchmark: GoEmotions text baseline on RAVDESS transcripts.
Evaluation only -- not part of the app.
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
PROJECT_ROOT = os.path.dirname(TESTS_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

from goemotion_classifier import load_goemotion_model, classify_emotion, EMOTION_MAP

DEFAULT_DATASET_DIR = "/Users/alizeaarif/Desktop/Audio_Speech_Actors_01-24"
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Emotion Detection from Speech", "GoEmotions_RAVDESS")

RAVDESS_CODE_NAMES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
# Same canonical RAVDESS -> UrbanPulse mapping used for every model tested this round.
RAVDESS_TO_URBANPULSE = {
    "01": "Calm", "02": "Calm", "03": "Content", "04": "Anxious",
    "05": "Stressed", "06": "Anxious", "07": "Annoyed", "08": "Calm",
}
URBANPULSE_STATES = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]

STATEMENT_TEXT = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


# Loads the evaluation clips and their labels.
def load_dataset(dataset_dir, sample):
    entries = []
    for actor_dir in sorted(os.listdir(dataset_dir)):
        full_actor_dir = os.path.join(dataset_dir, actor_dir)
        if not os.path.isdir(full_actor_dir):
            continue
        for fn in sorted(os.listdir(full_actor_dir)):
            if not fn.endswith(".wav"):
                continue
            parts = fn.replace(".wav", "").split("-")
            if len(parts) != 7:
                continue
            emotion_code = parts[2]
            statement_code = parts[4]
            if emotion_code not in RAVDESS_TO_URBANPULSE or statement_code not in STATEMENT_TEXT:
                continue
            true_state = RAVDESS_TO_URBANPULSE[emotion_code]
            entries.append((os.path.join(actor_dir, fn), true_state, emotion_code, statement_code))

    entries.sort(key=lambda e: e[0])
    if sample:
        import random
        random.seed(42)
        entries = random.sample(entries, min(sample, len(entries)))
        entries.sort(key=lambda e: e[0])

    return entries


# Runs the model over every clip and collects predictions and timings.
def run_evaluation(dataset_dir, sample):
    entries = load_dataset(dataset_dir, sample)
    print(f"Evaluating GoEmotions (text-based, production pipeline) on {len(entries)} RAVDESS clips, "
          f"using the known fixed sentence text (no transcription errors — best case for this approach).")

    load_goemotion_model()

    results = []
    t_start = time.time()

    for i, (rel_path, true_state, emotion_code, statement_code) in enumerate(entries, start=1):
        text = STATEMENT_TEXT[statement_code]

        t0 = time.time()
        result = classify_emotion(text)
        elapsed = time.time() - t0

        results.append({
            "filename": rel_path,
            "true_state": true_state,
            "true_ravdess_emotion": RAVDESS_CODE_NAMES[emotion_code],
            "text": text,
            "pred_state": result["emotion"],
            "raw_label": result["raw_label"],
            "raw_confidence": result["confidence"],
            "time_s": elapsed,
        })

        if i % 50 == 0 or i == len(entries):
            elapsed_total = time.time() - t_start
            rate = i / elapsed_total
            eta = (len(entries) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(entries)} clips | {elapsed_total:.0f}s elapsed | ETA {eta:.0f}s")

    return results


# Computes the headline metrics and writes metrics.txt.
def report_metrics(results):
    y_true = [r["true_state"] for r in results]
    y_pred = [r["pred_state"] for r in results]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=URBANPULSE_STATES, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=URBANPULSE_STATES, average="macro", zero_division=0
    )
    avg_time_s = sum(r["time_s"] for r in results) / len(results)

    cm = confusion_matrix(y_true, y_pred, labels=URBANPULSE_STATES)
    n_unique_preds = len(set(y_pred))

    lines = []
    lines.append("=" * 84)
    lines.append("GoEmotions (text-based, production Step 2 pipeline) on RAVDESS")
    lines.append("Step 3: Emotion Detection from Speech — baseline comparison")
    lines.append("=" * 84)
    lines.append("")
    lines.append("SUMMARY (UrbanPulse 5-state mapping)")
    lines.append("-" * 84)
    lines.append(f"Accuracy   : {accuracy:.4f}   ({accuracy*100:.2f}% of clips correctly classified)")
    lines.append(f"Precision  : {macro_p:.4f}   (macro — average trustworthiness across all 5 states)")
    lines.append(f"Recall     : {macro_r:.4f}   (macro — average how much it actually catches)")
    lines.append(f"F1         : {macro_f1:.4f}   (combined precision+recall score)")
    lines.append(f"Avg Time   : {avg_time_s*1000:.0f} ms per clip")
    lines.append(f"Clips evaluated: {len(results)}  (all 8 RAVDESS emotions usable — text is content-based, not audio-limited)")
    lines.append("-" * 84)
    lines.append(f"NOTE: GoEmotions sees only the FIXED, EMOTION-NEUTRAL SENTENCE TEXT (\"Kids/Dogs are "
                  f"talking/sitting by the door.\"), never the audio. Every clip of the same statement gets "
                  f"the exact SAME text, so GoEmotions predicts the SAME emotion for every actor and every "
                  f"true emotion sharing a statement — it produced only {n_unique_preds} distinct predicted "
                  f"state(s) across all {len(results)} clips. This is the best case for this approach (a real "
                  f"transcription model could only add error on top). The gap below the dedicated audio "
                  f"models (HuBERT/Yassmen/audEERING) is the expected, real finding: text content carries no "
                  f"information about how a sentence was actually spoken.")
    lines.append("-" * 84)
    lines.append("")
    lines.append("PER-CLASS BREAKDOWN")
    lines.append("-" * 84)
    lines.append(f"{'UrbanPulse State':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}")
    lines.append("-" * 84)
    for state, p, r, f, s in zip(URBANPULSE_STATES, precision, recall, f1, support):
        lines.append(f"{state:<20}{p:>12.3f}{r:>12.3f}{f:>12.3f}{s:>10d}")
    lines.append("-" * 84)
    lines.append("")
    lines.append("CONFUSION MATRIX (rows = true RAVDESS emotion mapped to UrbanPulse state, columns = predicted)")
    lines.append("-" * 84)
    header = f"{'True \\ Pred':<12}" + "".join(f"{s:>10}" for s in URBANPULSE_STATES)
    lines.append(header)
    for i, state in enumerate(URBANPULSE_STATES):
        row = f"{state:<12}" + "".join(f"{cm[i][j]:>10d}" for j in range(len(URBANPULSE_STATES)))
        lines.append(row)
    lines.append("-" * 84)
    lines.append("")
    lines.append("SAMPLE PREDICTIONS (first 10 clips, for qualitative inspection)")
    lines.append("-" * 84)
    for r in results[:10]:
        correct = "correct" if r["true_state"] == r["pred_state"] else "WRONG"
        lines.append(f"{r['filename']}  [{r['true_ravdess_emotion']} -> {r['true_state']}]  text=\"{r['text']}\"  "
                     f"predicted: {r['raw_label']} -> {r['pred_state']} (conf={r['raw_confidence']:.2f}) ({correct})")
    lines.append("")
    lines.append("=" * 84)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    return {
        "accuracy": accuracy, "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "avg_time_ms": avg_time_s * 1000, "n_clips": len(results),
        "per_class": {"precision": precision, "recall": recall, "f1": f1, "support": support},
        "confusion_matrix": cm,
        "report_text": report_text,
    }


# Writes the result charts used in the report.
def save_graphs(metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(URBANPULSE_STATES)), URBANPULSE_STATES, rotation=45, ha="right")
    plt.yticks(range(len(URBANPULSE_STATES)), URBANPULSE_STATES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("GoEmotions (text-based) on RAVDESS — Confusion Matrix\n(sees only fixed sentence text, not audio)")
    for i in range(len(URBANPULSE_STATES)):
        for j in range(len(URBANPULSE_STATES)):
            color = "white" if cm[i][j] > cm.max() / 2 else "black"
            plt.text(j, i, str(cm[i][j]), ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    per_class = metrics["per_class"]
    x = np.arange(len(URBANPULSE_STATES))
    width = 0.25
    plt.figure(figsize=(9, 6))
    plt.bar(x - width, per_class["precision"], width, label="Precision", color="#0F766E")
    plt.bar(x, per_class["recall"], width, label="Recall", color="#2563EB")
    plt.bar(x + width, per_class["f1"], width, label="F1", color="#B45309")
    plt.xticks(x, URBANPULSE_STATES)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("GoEmotions (text-based) on RAVDESS — Per-Class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_class_metrics.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["macro_p"], metrics["macro_r"], metrics["macro_f1"]]
    bars = plt.bar(labels, values, color="#9333EA")
    plt.ylim(0, 1.05)
    plt.title(f"GoEmotions (text-based) on RAVDESS — Overall Summary\n(Avg. time per clip: {metrics['avg_time_ms']:.0f} ms, n={metrics['n_clips']})")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "overall_summary.png"), dpi=150)
    plt.close()

    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(metrics["report_text"] + "\n")

    print(f"\nGraphs and metrics.txt saved to: {results_dir}")


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Evaluate GoEmotions (text-based) on RAVDESS")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    results = run_evaluation(args.dataset_dir, args.sample)
    metrics = report_metrics(results)
    save_graphs(metrics, RESULTS_DIR)


if __name__ == "__main__":
    main()
