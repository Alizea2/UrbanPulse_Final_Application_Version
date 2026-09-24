"""Step 3 benchmark: audEERING dimensional SER on RAVDESS.
Evaluation only -- not part of the app.
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
import torch.nn as nn
import librosa
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, AutoConfig, AutoProcessor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

DEFAULT_DATASET_DIR = "/Users/alizeaarif/Desktop/Audio_Speech_Actors_01-24"
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Emotion Detection from Speech", "Audeering_RAVDESS")

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SAMPLE_RATE = 16000

RAVDESS_CODE_NAMES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
# All 8 RAVDESS emotions have a valid circumplex position, so none need be excluded here.
# Expected (arousal, valence) quadrant per RAVDESS emotion, per the circumplex model of emotion.
RAVDESS_TO_QUADRANT = {
    "01": ("low", "mid"),     # neutral
    "02": ("low", "high"),    # calm
    "03": ("high", "high"),   # happy
    "04": ("low", "low"),     # sad
    "05": ("high", "low"),    # angry
    "06": ("high", "low"),    # fearful
    "07": ("mid", "low"),     # disgust
    "08": ("high", "mid"),    # surprised
}


# Maps the model's valence/arousal output onto the 5 wellbeing states.
def quadrant_to_urbanpulse(arousal_level, valence_level):
    if valence_level == "high":
        return "Content" if arousal_level != "low" else "Calm"
    if valence_level == "low":
        return "Stressed" if arousal_level == "high" else ("Anxious" if arousal_level == "mid" else "Anxious")
    return "Calm"


RAVDESS_TO_URBANPULSE = {
    code: quadrant_to_urbanpulse(a, v) for code, (a, v) in RAVDESS_TO_QUADRANT.items()
}
URBANPULSE_STATES = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        # NOTE: init_weights() is deliberately skipped — under the current transformers
        # version it triggers tie_weights() -> all_tied_weights_keys, an internal API
        # that no longer exists for this custom model class. Skipping it is safe here
        # because every weight this model needs is immediately overwritten by the
        # checkpoint's state dict in load_model() right after construction.

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# Loads the model under test.
def load_model():
    print(f"Loading {MODEL_NAME} (bypassing from_pretrained() due to a transformers version incompatibility)...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = EmotionModel(config)

    ckpt_path = hf_hub_download(MODEL_NAME, "pytorch_model.bin")
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint did not load cleanly — missing={missing}, unexpected={unexpected}")
    print("Checkpoint loaded cleanly (0 missing, 0 unexpected keys).")
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


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
            if emotion_code not in RAVDESS_TO_URBANPULSE:
                continue
            true_state = RAVDESS_TO_URBANPULSE[emotion_code]
            entries.append((os.path.join(actor_dir, fn), true_state, emotion_code))

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
    print(f"Evaluating audEERING dimensional model on {len(entries)} RAVDESS clips "
          f"(a corpus it was never trained on).")

    model, processor = load_model()

    results = []
    t_start = time.time()

    for i, (rel_path, true_state, ravdess_code) in enumerate(entries, start=1):
        wav_path = os.path.join(dataset_dir, rel_path)
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

        t0 = time.time()
        inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            _, logits = model(inputs.input_values)
        arousal, dominance, valence = [float(v) for v in logits[0].numpy()]
        elapsed = time.time() - t0

        results.append({
            "filename": rel_path,
            "true_state": true_state,
            "true_ravdess_emotion": RAVDESS_CODE_NAMES[ravdess_code],
            "arousal": arousal,
            "dominance": dominance,
            "valence": valence,
            "time_s": elapsed,
        })

        if i % 50 == 0 or i == len(entries):
            elapsed_total = time.time() - t_start
            rate = i / elapsed_total
            eta = (len(entries) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(entries)} clips | {elapsed_total:.0f}s elapsed | ETA {eta:.0f}s")

    # Median-split calibration on this run's own arousal/valence distributions
    arousals = np.array([r["arousal"] for r in results])
    valences = np.array([r["valence"] for r in results])
    a_low, a_high = np.percentile(arousals, [33.3, 66.7])
    v_low, v_high = np.percentile(valences, [33.3, 66.7])

    def bucket(value, low, high):
        if value <= low:
            return "low"
        if value >= high:
            return "high"
        return "mid"

    for r in results:
        a_level = bucket(r["arousal"], a_low, a_high)
        v_level = bucket(r["valence"], v_low, v_high)
        r["pred_state"] = quadrant_to_urbanpulse(a_level, v_level)

    return results, (a_low, a_high, v_low, v_high)


# Computes the headline metrics and writes metrics.txt.
def report_metrics(results, thresholds):
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
    a_low, a_high, v_low, v_high = thresholds

    lines = []
    lines.append("=" * 84)
    lines.append("audEERING dimensional model (arousal/dominance/valence) on RAVDESS")
    lines.append("Step 3: Emotion Detection from Speech")
    lines.append("=" * 84)
    lines.append("")
    lines.append("SUMMARY (UrbanPulse 5-state mapping, via median-split circumplex classification)")
    lines.append("-" * 84)
    lines.append(f"Accuracy   : {accuracy:.4f}   ({accuracy*100:.2f}% of clips correctly classified)")
    lines.append(f"Precision  : {macro_p:.4f}   (macro — average trustworthiness across all 5 states)")
    lines.append(f"Recall     : {macro_r:.4f}   (macro — average how much it actually catches)")
    lines.append(f"F1         : {macro_f1:.4f}   (combined precision+recall score)")
    lines.append(f"Avg Time   : {avg_time_s*1000:.0f} ms per clip")
    lines.append(f"Clips evaluated: {len(results)}  (all 8 RAVDESS emotions usable — dimensional output has no gaps)")
    lines.append("-" * 84)
    lines.append(f"Calibration thresholds (percentile-based, computed from this run's own data):")
    lines.append(f"  Arousal: low <= {a_low:.3f}, high >= {a_high:.3f}")
    lines.append(f"  Valence: low <= {v_low:.3f}, high >= {v_high:.3f}")
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
                     f"predicted: {r['pred_state']} (A={r['arousal']:.2f}, V={r['valence']:.2f}) ({correct})")
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
    plt.title("audEERING dimensional model on RAVDESS — Confusion Matrix\n(never trained on this dataset)")
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
    plt.title("audEERING dimensional model on RAVDESS — Per-Class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_class_metrics.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["macro_p"], metrics["macro_r"], metrics["macro_f1"]]
    bars = plt.bar(labels, values, color="#9333EA")
    plt.ylim(0, 1.05)
    plt.title(f"audEERING dimensional model on RAVDESS — Overall Summary\n(Avg. time per clip: {metrics['avg_time_ms']:.0f} ms, n={metrics['n_clips']})")
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
    parser = argparse.ArgumentParser(description="Evaluate audEERING dimensional model on RAVDESS")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    results, thresholds = run_evaluation(args.dataset_dir, args.sample)
    metrics = report_metrics(results, thresholds)
    save_graphs(metrics, RESULTS_DIR)


if __name__ == "__main__":
    main()
