"""Step 1 benchmark: EfficientAT (mn10_as) on SONYC-UST's 23 fine-grained
labels. Evaluation only -- this is the model the app went on to use.
"""

import argparse
import contextlib
import csv
import io
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
import torch
import librosa
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Tests/model_evaluation/Sound Classification
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../Tests
DEFAULT_DATASET_DIR = "/Users/alizeaarif/Desktop/SONYC"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Test-Results", "Sound Classification", "EfficientAT_SONYC-UST")
EFFICIENTAT_REPO = "/tmp/EfficientAT"  # local clone of github.com/fschmid56/EfficientAT

EFFICIENTAT_SAMPLE_RATE = 32000
EFFICIENTAT_MODEL_NAME = "mn10_as"  # standard/default efficient variant (width_mult=1.0)

FINE_TAXONOMY = [
    ("1-1", "small-sounding-engine"), ("1-2", "medium-sounding-engine"), ("1-3", "large-sounding-engine"),
    ("2-1", "rock-drill"), ("2-2", "jackhammer"), ("2-3", "hoe-ram"), ("2-4", "pile-driver"),
    ("3-1", "non-machinery-impact"),
    ("4-1", "chainsaw"), ("4-2", "small-medium-rotating-saw"), ("4-3", "large-rotating-saw"),
    ("5-1", "car-horn"), ("5-2", "car-alarm"), ("5-3", "siren"), ("5-4", "reverse-beeper"),
    ("6-1", "stationary-music"), ("6-2", "mobile-music"), ("6-3", "ice-cream-truck"),
    ("7-1", "person-or-small-group-talking"), ("7-2", "person-or-small-group-shouting"),
    ("7-3", "large-crowd"), ("7-4", "amplified-speech"),
    ("8-1", "dog-barking-whining"),
]
FINE_CLASSES = [name for _, name in FINE_TAXONOMY]
PRESENCE_COLUMNS = {name: f"{code}_{name}_presence" for code, name in FINE_TAXONOMY}

# Identical mapping to test_panns_sonyc_ust.py / test_epanns_sonyc_ust.py — same 527-class AudioSet vocabulary.
EFFICIENTAT_TO_SONYC_FINE = {}
UNMAPPABLE_CLASSES = ["hoe-ram", "pile-driver", "large-rotating-saw", "stationary-music", "mobile-music", "amplified-speech"]


# Records which of the model's labels map onto a dataset class.
def _register(fine_class, model_labels):
    for label in model_labels:
        EFFICIENTAT_TO_SONYC_FINE[label] = fine_class


_register("small-sounding-engine", ["Light engine (high frequency)"])
_register("medium-sounding-engine", ["Medium engine (mid frequency)"])
_register("large-sounding-engine", ["Heavy engine (low frequency)"])
_register("rock-drill", ["Drill"])
_register("jackhammer", ["Jackhammer"])
_register("non-machinery-impact", ["Bang", "Slam", "Knock", "Tap", "Clatter", "Thump, thud", "Smash, crash"])
_register("chainsaw", ["Chainsaw"])
_register("small-medium-rotating-saw", ["Sawing"])
_register("car-horn", ["Vehicle horn, car horn, honking", "Honk", "Toot"])
_register("car-alarm", ["Car alarm"])
_register("siren", ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)"])
_register("reverse-beeper", ["Reversing beeps"])
_register("ice-cream-truck", ["Ice cream truck, ice cream van"])
_register("person-or-small-group-talking", ["Speech", "Conversation", "Narration, monologue", "Child speech, kid speaking"])
_register("person-or-small-group-shouting", ["Shout", "Yell", "Children shouting", "Screaming", "Bellow", "Whoop"])
_register("large-crowd", ["Crowd", "Hubbub, speech noise, speech babble", "Chatter"])
_register("dog-barking-whining", ["Dog", "Bark", "Bow-wow", "Yip", "Whimper (dog)", "Growling"])

MIN_SAMPLES_SEC = 1.0


# Loads the model under test.
def load_efficientat():
    print("Loading EfficientAT (mn10_as, CPU)...")
    # helpers/utils.py and models/mn/model.py load their label CSV / checkpoints via
    # paths relative to the repo root, so we cd into the clone before importing.
    sys.path.insert(0, EFFICIENTAT_REPO)
    original_cwd = os.getcwd()
    os.chdir(EFFICIENTAT_REPO)
    try:
        from models.mn.model import get_model as get_mobilenet
        from models.preprocess import AugmentMelSTFT
        from helpers.utils import labels

        with contextlib.redirect_stdout(io.StringIO()):  # suppress the model's architecture dump
            model = get_mobilenet(width_mult=1.0, pretrained_name=EFFICIENTAT_MODEL_NAME)
        model.to("cpu")
        model.eval()

        mel = AugmentMelSTFT(n_mels=128, sr=EFFICIENTAT_SAMPLE_RATE, win_length=800, hopsize=320)
        mel.to("cpu")
        mel.eval()
    finally:
        os.chdir(original_cwd)

    return model, mel, labels


# Predicts one clip, mapping the model's own labels onto the dataset's.
def predict_topk_fine(model, mel, class_names, class_name_to_idx, wav_path, k):
    y, sr = librosa.load(wav_path, sr=EFFICIENTAT_SAMPLE_RATE, mono=True)
    min_samples = int(sr * MIN_SAMPLES_SEC)
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)))

    with torch.no_grad():
        waveform = torch.from_numpy(y[None, :]).float()
        spec = mel(waveform)
        preds, _ = model(spec.unsqueeze(0))
        scores = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    top_idx = np.argsort(scores)[::-1][:k]

    predicted = set()
    for i in top_idx:
        fine = EFFICIENTAT_TO_SONYC_FINE.get(class_names[i])
        if fine:
            predicted.add(fine)

    class_scores = {cls: 0.0 for cls in FINE_CLASSES}
    for label, cls in EFFICIENTAT_TO_SONYC_FINE.items():
        idx = class_name_to_idx.get(label)
        if idx is not None:
            class_scores[cls] = max(class_scores[cls], float(scores[idx]))

    return predicted, class_scores


# Builds one ground-truth label per clip from the annotations.
def load_ground_truth(dataset_dir, split):
    meta_path = os.path.join(dataset_dir, "annotations.csv")
    votes = defaultdict(lambda: defaultdict(list))
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            fn = row["audio_filename"]
            for cls, col in PRESENCE_COLUMNS.items():
                v = int(row[col])
                if v in (0, 1):
                    votes[fn][cls].append(v)

    ground_truth = {}
    for fn, class_votes in votes.items():
        labels_present = set()
        for cls in FINE_CLASSES:
            vs = class_votes.get(cls, [])
            if vs and (sum(vs) / len(vs)) > 0.5:
                labels_present.add(cls)
        ground_truth[fn] = labels_present
    return ground_truth


# Runs the model over every clip and collects predictions and timings.
def run_evaluation(dataset_dir, split, top_k):
    ground_truth = load_ground_truth(dataset_dir, split)
    print(f"Evaluating EfficientAT (top-{top_k}) on {len(ground_truth)} clips from the '{split}' split "
          f"across {len(FINE_CLASSES)} SONYC-UST fine-grained classes.")
    print(f"({len(UNMAPPABLE_CLASSES)} of these have no possible mapping: {', '.join(UNMAPPABLE_CLASSES)})")

    print("Indexing audio files...")
    file_index = {}
    for entry in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, entry)
        if entry.startswith("audio-") and os.path.isdir(full):
            for fn in os.listdir(full):
                file_index[fn] = os.path.join(full, fn)

    model, mel, class_names = load_efficientat()
    class_name_to_idx = {name: i for i, name in enumerate(class_names)}

    y_true, y_pred, y_score = [], [], []
    filenames = sorted(ground_truth.keys())
    skipped = 0
    t_start = time.time()
    inference_times = []

    for i, fn in enumerate(filenames, start=1):
        wav_path = file_index.get(fn)
        if wav_path is None:
            skipped += 1
            continue
        try:
            t0 = time.time()
            predicted, class_scores = predict_topk_fine(model, mel, class_names, class_name_to_idx, wav_path, top_k)
            inference_times.append(time.time() - t0)
        except Exception as e:
            print(f"  [skip] {fn}: {e}")
            skipped += 1
            continue

        true_labels = ground_truth[fn]
        y_true.append([1 if c in true_labels else 0 for c in FINE_CLASSES])
        y_pred.append([1 if c in predicted else 0 for c in FINE_CLASSES])
        y_score.append([class_scores[c] for c in FINE_CLASSES])

        if i % 50 == 0 or i == len(filenames):
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(filenames) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(filenames)} clips | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    print(f"Done. {skipped} clip(s) skipped (missing file or load error).")
    avg_time_ms = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0.0
    return np.array(y_true), np.array(y_pred), np.array(y_score), avg_time_ms


# AUC uses the raw confidences, so it ignores the top-k cutoff.
def compute_auc_per_class(y_true, y_score, classes):
    aucs = {}
    for i, cls in enumerate(classes):
        col_true = y_true[:, i]
        col_score = y_score[:, i]
        has_both_classes = len(set(col_true.tolist())) == 2
        has_signal = col_score.std() > 0
        if has_both_classes and has_signal:
            aucs[cls] = roc_auc_score(col_true, col_score)
        else:
            aucs[cls] = None
    return aucs


# Computes the headline metrics and writes metrics.txt.
def report_metrics(y_true, y_pred, y_score, avg_time_ms, top_k):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    hamming_acc = 1 - hamming_loss(y_true, y_pred)

    per_class_auc = compute_auc_per_class(y_true, y_score, FINE_CLASSES)
    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else None

    lines = []
    lines.append("=" * 84)
    lines.append(f"EfficientAT (mn10_as, top-{top_k}) on SONYC-UST — Step 1: Sound Classification (23 fine-grained classes)")
    lines.append("=" * 84)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 84)
    lines.append(f"Accuracy   : {hamming_acc:.3f}   (correct yes/no decisions, out of all checked)")
    lines.append(f"Precision  : {macro_p:.3f}   (average trustworthiness across all sound types)")
    lines.append(f"Recall     : {macro_r:.3f}   (average how much it actually catches)")
    lines.append(f"F1         : {macro_f1:.3f}   (combined precision+recall score)")
    lines.append(f"AUC        : {macro_auc:.3f}   (how well it ranks \"present\" above \"absent\", any cutoff)" if macro_auc is not None
                  else "AUC        : N/A     (not enough data to compute)")
    lines.append(f"Avg Time   : {avg_time_ms:.0f} ms per clip   (how long one classification took, on average)")
    lines.append("-" * 84)
    lines.append("")
    lines.append("PER-CLASS BREAKDOWN")
    lines.append("-" * 84)
    lines.append(f"{'Class':<34}{'Precision':>12}{'Recall':>12}{'F1':>12}{'AUC':>10}{'Support':>8}")
    lines.append("-" * 84)
    for cls, p, r, f, s in zip(FINE_CLASSES, precision, recall, f1, support):
        auc_val = per_class_auc[cls]
        auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"
        flag = "  [no mapping]" if cls in UNMAPPABLE_CLASSES else ""
        lines.append(f"{cls:<34}{p:>12.3f}{r:>12.3f}{f:>12.3f}{auc_str:>10}{s:>8d}{flag}")
    lines.append("=" * 84)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    return {
        "hamming_acc": hamming_acc,
        "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "macro_auc": macro_auc, "avg_time_ms": avg_time_ms,
        "per_class": {"precision": precision, "recall": recall, "f1": f1, "support": support},
        "report_text": report_text,
    }


# Writes the result charts used in the report.
def save_graphs(y_true, y_pred, metrics, results_dir, top_k):
    os.makedirs(results_dir, exist_ok=True)
    per_class = metrics["per_class"]
    x = np.arange(len(FINE_CLASSES))
    width = 0.25
    labels_x = [f"{c}*" if c in UNMAPPABLE_CLASSES else c for c in FINE_CLASSES]

    plt.figure(figsize=(16, 7))
    plt.bar(x - width, per_class["precision"], width, label="Precision", color="#0F766E")
    plt.bar(x, per_class["recall"], width, label="Recall", color="#2563EB")
    plt.bar(x + width, per_class["f1"], width, label="F1", color="#B45309")
    plt.xticks(x, labels_x, rotation=60, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"EfficientAT (mn10_as, top-{top_k}) on SONYC-UST — Fine-Grained Precision / Recall / F1\n(* = no possible mapping)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_class_metrics.png"), dpi=150)
    plt.close()

    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    plt.figure(figsize=(16, 7))
    plt.bar(x - width, tp, width, label="True Positive", color="#16A34A")
    plt.bar(x, fp, width, label="False Positive", color="#DC2626")
    plt.bar(x + width, fn, width, label="False Negative", color="#CA8A04")
    plt.xticks(x, labels_x, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Count")
    plt.title(f"EfficientAT (mn10_as, top-{top_k}) on SONYC-UST — Fine-Grained TP / FP / FN\n(* = no possible mapping)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tp_fp_fn.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    overall_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    auc_val = metrics["macro_auc"] if metrics["macro_auc"] is not None else 0.0
    overall_values = [metrics["hamming_acc"], metrics["macro_p"], metrics["macro_r"], metrics["macro_f1"], auc_val]
    bars = plt.bar(overall_labels, overall_values, color="#DB2777")
    plt.ylim(0, 1.05)
    plt.title(f"EfficientAT (mn10_as, top-{top_k}) on SONYC-UST — Overall Summary\n(Avg. time per clip: {metrics['avg_time_ms']:.0f} ms)")
    for bar, val in zip(bars, overall_values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "overall_summary.png"), dpi=150)
    plt.close()

    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(metrics["report_text"] + "\n")

    print(f"\nGraphs and metrics.txt saved to: {results_dir}")


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientAT on SONYC-UST's 23 fine-grained classes")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default="test", choices=["test", "validate", "train"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    y_true, y_pred, y_score, avg_time_ms = run_evaluation(args.dataset_dir, args.split, args.top_k)
    metrics = report_metrics(y_true, y_pred, y_score, avg_time_ms, args.top_k)
    save_graphs(y_true, y_pred, metrics, RESULTS_DIR, args.top_k)


if __name__ == "__main__":
    main()
