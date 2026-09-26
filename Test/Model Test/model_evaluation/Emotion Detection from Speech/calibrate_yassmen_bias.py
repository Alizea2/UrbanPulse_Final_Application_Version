"""Sweeps the FEA/SAD logit offset for the Yassmen SER model on RAVDESS to
pick the value used in models/emotion_classifier.py.
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

DEFAULT_DATASET_DIR = os.path.join(TESTS_ROOT, "Datasets", "Emotion Detection from Speech", "RAVDESS_Audio_Speech_Actors_01-24")
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Emotion Detection from Speech", "Yassmen_Calibration")

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
RAVDESS_TO_MODEL_LABEL = {"01": "NEU", "03": "HAP", "04": "SAD", "05": "ANG", "06": "FEA", "07": "DIS"}
MODEL_LABEL_TO_URBANPULSE = {
    "ANG": "Stressed", "DIS": "Annoyed", "FEA": "Anxious",
    "SAD": "Anxious", "HAP": "Content", "NEU": "Calm",
}
URBANPULSE_STATES = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]

# Candidate bias values (subtracted from FEA and SAD logits before
# argmax). 0.0 is the unmodified baseline, included for direct comparison.
CANDIDATE_BIASES = [0.0, -0.3, -0.5, -0.7, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0]


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
            if emotion_code in ("02", "08") or emotion_code not in RAVDESS_TO_MODEL_LABEL:
                continue
            model_label = RAVDESS_TO_MODEL_LABEL[emotion_code]
            true_state = MODEL_LABEL_TO_URBANPULSE[model_label]
            entries.append((os.path.join(actor_dir, fn), true_state, model_label, emotion_code))

    entries.sort(key=lambda e: e[0])
    if sample:
        import random
        random.seed(42)
        entries = random.sample(entries, min(sample, len(entries)))
        entries.sort(key=lambda e: e[0])
    return entries


def run_inference_once(dataset_dir, sample):
    """Runs the model once per clip, caching raw logits — the sweep
    below reuses this instead of re-running inference per bias value."""
    entries = load_dataset(dataset_dir, sample)
    print(f"Running inference once on {len(entries)} clips (logits cached for the sweep)...")

    model, extractor = load_model()
    label2id = model.config.label2id
    id2label = model.config.id2label
    print(f"Label order: {id2label}")

    cached = []
    t_start = time.time()
    for i, (rel_path, true_state, true_model_label, ravdess_code) in enumerate(entries, start=1):
        wav_path = os.path.join(dataset_dir, rel_path)
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        inputs = extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits[0].numpy()
        cached.append({
            "filename": rel_path,
            "true_state": true_state,
            "true_ravdess_emotion": RAVDESS_CODE_NAMES[ravdess_code],
            "logits": logits,
        })
        if i % 50 == 0 or i == len(entries):
            elapsed_total = time.time() - t_start
            rate = i / elapsed_total
            eta = (len(entries) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(entries)} clips | {elapsed_total:.0f}s elapsed | ETA {eta:.0f}s")

    return cached, label2id, id2label


# Scores the eval set once at a single candidate bias value.
def evaluate_bias(cached, label2id, id2label, bias):
    fea_idx = label2id["FEA"]
    sad_idx = label2id["SAD"]

    y_true, y_pred = [], []
    for entry in cached:
        logits = entry["logits"].copy()
        logits[fea_idx] += bias
        logits[sad_idx] += bias
        top_idx = int(np.argmax(logits))
        raw_label = id2label[top_idx]
        pred_state = MODEL_LABEL_TO_URBANPULSE[raw_label]
        y_true.append(entry["true_state"])
        y_pred.append(pred_state)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=URBANPULSE_STATES, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=URBANPULSE_STATES, average="macro", zero_division=0
    )
    pred_counts = {s: y_pred.count(s) for s in URBANPULSE_STATES}

    return {
        "bias": bias, "accuracy": accuracy, "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "per_class": {"precision": precision, "recall": recall, "f1": f1, "support": support},
        "pred_counts": pred_counts,
    }


# Reports the whole sweep and picks the best-scoring bias.
def report_sweep(sweep_results):
    lines = []
    lines.append("=" * 100)
    lines.append("Yassmen FEA/SAD Logit-Bias Calibration Sweep — RAVDESS — Step 3: Emotion Detection")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Why: production usage showed predictions skewing toward 'Anxious', which receives BOTH")
    lines.append("FEA and SAD (Yassmen's two native labels covering fear/sadness), giving it structurally")
    lines.append("double the chance of being the argmax vs. every other UrbanPulse state (single label each).")
    lines.append("")
    lines.append("-" * 100)
    lines.append(f"{'Bias':>8}{'Accuracy':>12}{'Macro P':>12}{'Macro R':>12}{'Macro F1':>12}   Prediction distribution")
    lines.append("-" * 100)
    for r in sweep_results:
        dist = ", ".join(f"{s}={r['pred_counts'][s]}" for s in URBANPULSE_STATES)
        lines.append(f"{r['bias']:>8.2f}{r['accuracy']:>12.4f}{r['macro_p']:>12.4f}{r['macro_r']:>12.4f}{r['macro_f1']:>12.4f}   {dist}")
    lines.append("-" * 100)
    lines.append("")

    best = max(sweep_results, key=lambda r: r["macro_f1"])
    lines.append(f"BEST by macro F1: bias={best['bias']:.2f}  (macro F1={best['macro_f1']:.4f}, "
                 f"vs baseline bias=0.00 macro F1={sweep_results[0]['macro_f1']:.4f})")
    lines.append("")
    lines.append("PER-CLASS BREAKDOWN AT BEST BIAS")
    lines.append("-" * 100)
    lines.append(f"{'UrbanPulse State':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}")
    lines.append("-" * 100)
    for state, p, r, f, s in zip(URBANPULSE_STATES, best["per_class"]["precision"], best["per_class"]["recall"],
                                  best["per_class"]["f1"], best["per_class"]["support"]):
        lines.append(f"{state:<20}{p:>12.3f}{r:>12.3f}{f:>12.3f}{s:>10d}")
    lines.append("-" * 100)
    lines.append("")
    lines.append("PER-CLASS BREAKDOWN AT BASELINE (bias=0.00, current production behavior)")
    lines.append("-" * 100)
    lines.append(f"{'UrbanPulse State':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}")
    lines.append("-" * 100)
    baseline = sweep_results[0]
    for state, p, r, f, s in zip(URBANPULSE_STATES, baseline["per_class"]["precision"], baseline["per_class"]["recall"],
                                  baseline["per_class"]["f1"], baseline["per_class"]["support"]):
        lines.append(f"{state:<20}{p:>12.3f}{r:>12.3f}{f:>12.3f}{s:>10d}")
    lines.append("=" * 100)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    return report_text, best


# Writes the result chart used in the report.
def save_graph(sweep_results, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    biases = [r["bias"] for r in sweep_results]

    plt.figure(figsize=(10, 6))
    plt.plot(biases, [r["macro_f1"] for r in sweep_results], marker="o", label="Macro F1", color="#0F766E")
    plt.plot(biases, [r["macro_p"] for r in sweep_results], marker="o", label="Macro Precision", color="#7C3AED")
    plt.plot(biases, [r["macro_r"] for r in sweep_results], marker="o", label="Macro Recall", color="#DB2777")
    plt.xlabel("FEA/SAD logit bias (0 = unmodified baseline)")
    plt.ylabel("Score")
    plt.title("Yassmen Calibration Sweep — Macro Metrics vs. FEA/SAD Logit Bias")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bias_sweep.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    width = 0.8 / len(URBANPULSE_STATES)
    x = np.arange(len(biases))
    for i, state in enumerate(URBANPULSE_STATES):
        counts = [r["pred_counts"][state] for r in sweep_results]
        offset = (i - (len(URBANPULSE_STATES) - 1) / 2) * width
        plt.bar(x + offset, counts, width, label=state)
    plt.xticks(x, [f"{b:.2f}" for b in biases])
    plt.xlabel("FEA/SAD logit bias")
    plt.ylabel("Number of clips predicted as this state")
    plt.title("Prediction Distribution Shift Across the Bias Sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prediction_distribution.png"), dpi=150)
    plt.close()


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Calibration sweep for Yassmen FEA/SAD logit bias")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    cached, label2id, id2label = run_inference_once(args.dataset_dir, args.sample)

    sweep_results = [evaluate_bias(cached, label2id, id2label, bias) for bias in CANDIDATE_BIASES]

    report_text, best = report_sweep(sweep_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    save_graph(sweep_results, RESULTS_DIR)
    print(f"\nCalibration sweep saved to: {RESULTS_DIR}")
    print(f"Recommended bias for production: {best['bias']:.2f}")


if __name__ == "__main__":
    main()
