"""Step 3 benchmark: Yassmen wav2vec2 (CREMA-D fine-tuned) on RAVDESS.
Evaluation only -- this is the model the app went on to use.
"""

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Wav2Vec2ForSequenceClassification, AutoConfig, AutoFeatureExtractor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Tests/model_evaluation/Emotion Detection from Speech
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../Tests

DEFAULT_DATASET_DIR = os.path.join(TESTS_ROOT, "Datasets", "Emotion Detection from Speech", "RAVDESS_Audio_Speech_Actors_01-24")
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Emotion Detection from Speech", "Yassmen_RAVDESS")

MODEL_NAME = "Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition"
SAMPLE_RATE = 16000

KEY_RENAME = {
    "classifier.dense.weight": "projector.weight",
    "classifier.dense.bias": "projector.bias",
    "classifier.out_proj.weight": "classifier.weight",
    "classifier.out_proj.bias": "classifier.bias",
}

RAVDESS_CODE_NAMES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
# Only codes with a direct equivalent in Yassmen's 6-label (CREMA-D-shaped) output space.
RAVDESS_TO_MODEL_LABEL = {"01": "NEU", "03": "HAP", "04": "SAD", "05": "ANG", "06": "FEA", "07": "DIS"}
MODEL_LABEL_TO_URBANPULSE = {
    "ANG": "Stressed", "DIS": "Annoyed", "FEA": "Anxious",
    "SAD": "Anxious", "HAP": "Content", "NEU": "Calm",
}
URBANPULSE_STATES = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]


# Loads the model under test.
def load_model():
    print(f"Loading {MODEL_NAME} (with corrected checkpoint key mapping)...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.classifier_proj_size = 1024
    model = Wav2Vec2ForSequenceClassification(config)

    path = hf_hub_download(MODEL_NAME, "model.safetensors")
    sd = load_file(path)
    fixed_sd = {KEY_RENAME.get(k, k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(fixed_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint did not load cleanly — missing={missing}, unexpected={unexpected}")
    print("Checkpoint loaded cleanly (0 missing, 0 unexpected keys).")
    model.eval()

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, extractor


# Loads the evaluation clips and their labels.
def load_dataset(dataset_dir, sample):
    entries = []
    excluded = 0
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
            if emotion_code in ("02", "08"):
                excluded += 1
                continue
            if emotion_code not in RAVDESS_TO_MODEL_LABEL:
                continue
            model_label = RAVDESS_TO_MODEL_LABEL[emotion_code]
            true_state = MODEL_LABEL_TO_URBANPULSE[model_label]
            entries.append((os.path.join(actor_dir, fn), true_state, model_label, emotion_code))

    print(f"Excluded {excluded} calm/surprised clips (no equivalent in any Step 3 model tested this round).")
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
    print(f"Evaluating Yassmen wav2vec2 (CREMA-D fine-tuned) on {len(entries)} RAVDESS clips "
          f"(a corpus it was never trained on).")

    model, extractor = load_model()

    results = []
    t_start = time.time()

    for i, (rel_path, true_state, true_model_label, ravdess_code) in enumerate(entries, start=1):
        wav_path = os.path.join(dataset_dir, rel_path)
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

        t0 = time.time()
        inputs = extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        elapsed = time.time() - t0

        top_idx = int(np.argmax(probs))
        raw_label = model.config.id2label[top_idx]
        pred_state = MODEL_LABEL_TO_URBANPULSE[raw_label]

        results.append({
            "filename": rel_path,
            "true_state": true_state,
            "true_ravdess_emotion": RAVDESS_CODE_NAMES[ravdess_code],
            "pred_state": pred_state,
            "raw_label": raw_label,
            "raw_confidence": float(probs[top_idx]),
            "native_match": raw_label == true_model_label,
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
    native_accuracy = sum(1 for r in results if r["native_match"]) / len(results)

    cm = confusion_matrix(y_true, y_pred, labels=URBANPULSE_STATES)

    lines = []
    lines.append("=" * 84)
    lines.append(f"Yassmen wav2vec2 (CREMA-D fine-tuned) on RAVDESS — Step 3: Emotion Detection from Speech")
    lines.append("=" * 84)
    lines.append("")
    lines.append("SUMMARY (UrbanPulse 5-state mapping)")
    lines.append("-" * 84)
    lines.append(f"Accuracy   : {accuracy:.4f}   ({accuracy*100:.2f}% of clips correctly classified)")
    lines.append(f"Precision  : {macro_p:.4f}   (macro — average trustworthiness across all 5 states)")
    lines.append(f"Recall     : {macro_r:.4f}   (macro — average how much it actually catches)")
    lines.append(f"F1         : {macro_f1:.4f}   (combined precision+recall score)")
    lines.append(f"Avg Time   : {avg_time_s*1000:.0f} ms per clip")
    lines.append(f"Clips evaluated: {len(results)}  (calm/surprised excluded — no model equivalent)")
    lines.append("-" * 84)
    lines.append(f"Native 6-class accuracy (model's raw label vs RAVDESS ground truth, before UrbanPulse "
                 f"mapping): {native_accuracy:.4f} ({native_accuracy*100:.2f}%)")
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
        lines.append(f"{r['filename']}  [{r['true_ravdess_emotion']} -> {r['true_state']}]  "
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
    plt.title("Yassmen wav2vec2 (CREMA-D fine-tuned) on RAVDESS — Confusion Matrix\n(never trained on this dataset)")
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
    plt.title("Yassmen wav2vec2 on RAVDESS — Per-Class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_class_metrics.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["macro_p"], metrics["macro_r"], metrics["macro_f1"]]
    bars = plt.bar(labels, values, color="#9333EA")
    plt.ylim(0, 1.05)
    plt.title(f"Yassmen wav2vec2 on RAVDESS — Overall Summary\n(Avg. time per clip: {metrics['avg_time_ms']:.0f} ms, n={metrics['n_clips']})")
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
    parser = argparse.ArgumentParser(description="Evaluate Yassmen (CREMA-D fine-tuned) on RAVDESS")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    results = run_evaluation(args.dataset_dir, args.sample)
    metrics = report_metrics(results)
    save_graphs(metrics, RESULTS_DIR)


if __name__ == "__main__":
    main()
