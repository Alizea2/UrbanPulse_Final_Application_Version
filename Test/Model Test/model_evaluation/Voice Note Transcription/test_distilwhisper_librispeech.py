"""Step 2 benchmark: Distil-Whisper on LibriSpeech test-clean.
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
import librosa
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import EnglishTextNormalizer
import jiwer

transformers.logging.set_verbosity_error()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Tests/model_evaluation/Voice Note Transcription
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../Tests

DEFAULT_DATASET_DIR = os.path.join(TESTS_ROOT, "Datasets", "Voice Note Transcription", "LibriSpeech_test-clean")
RESULTS_DIR = os.path.join(TESTS_ROOT, "Test-Results", "Voice Note Transcription", "DistilWhisper_LibriSpeech")

MODEL_NAME = "distil-whisper/distil-small.en"
SAMPLE_RATE = 16000


# Loads the reference transcripts the predictions are scored against.
def load_references(dataset_dir):
    references = {}
    for speaker in sorted(os.listdir(dataset_dir)):
        speaker_dir = os.path.join(dataset_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for chapter in sorted(os.listdir(speaker_dir)):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if not os.path.isdir(chapter_dir):
                continue
            trans_path = os.path.join(chapter_dir, f"{speaker}-{chapter}.trans.txt")
            if not os.path.exists(trans_path):
                continue
            with open(trans_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, text = line.split(" ", 1)
                    flac_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                    if os.path.exists(flac_path):
                        references[utt_id] = (text, flac_path)
    return references


# Runs the model on one audio file and returns its transcript.
def transcribe(model, processor, flac_path):
    y, sr = librosa.load(flac_path, sr=SAMPLE_RATE, mono=True)
    inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features, max_new_tokens=256)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


# Runs the model over every clip and collects predictions and timings.
def run_evaluation(dataset_dir, sample):
    references = load_references(dataset_dir)
    utt_ids = sorted(references.keys())
    if sample:
        import random
        random.seed(42)  # same seed as the other two scripts -> identical clip subset
        utt_ids = sorted(random.sample(utt_ids, min(sample, len(utt_ids))))

    print(f"Evaluating Distil-Whisper ({MODEL_NAME}) on {len(utt_ids)} clips from LibriSpeech test-clean.")
    print(f"Loading {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    normalizer = EnglishTextNormalizer()

    results = []
    t_start = time.time()

    for i, utt_id in enumerate(utt_ids, start=1):
        reference_raw, flac_path = references[utt_id]
        t0 = time.time()
        hypothesis_raw = transcribe(model, processor, flac_path)
        elapsed = time.time() - t0

        ref_norm = normalizer(reference_raw)
        hyp_norm = normalizer(hypothesis_raw)
        if not ref_norm.strip():
            ref_norm = "<empty>"
        if not hyp_norm.strip():
            hyp_norm = "<empty>"

        results.append({
            "utt_id": utt_id,
            "reference_raw": reference_raw,
            "hypothesis_raw": hypothesis_raw,
            "reference_norm": ref_norm,
            "hypothesis_norm": hyp_norm,
            "time_s": elapsed,
        })

        if i % 100 == 0 or i == len(utt_ids):
            elapsed_total = time.time() - t_start
            rate = i / elapsed_total
            eta = (len(utt_ids) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(utt_ids)} clips | {elapsed_total:.0f}s elapsed | ETA {eta:.0f}s")

    return results


# Computes the headline metrics and writes metrics.txt.
def report_metrics(results):
    refs = [r["reference_norm"] for r in results]
    hyps = [r["hypothesis_norm"] for r in results]

    overall_wer = jiwer.wer(refs, hyps)
    overall_cer = jiwer.cer(refs, hyps)
    avg_time_s = sum(r["time_s"] for r in results) / len(results)

    per_clip_wer = [jiwer.wer(r["reference_norm"], r["hypothesis_norm"]) for r in results]
    perfect_clips = sum(1 for w in per_clip_wer if w == 0.0)

    lines = []
    lines.append("=" * 84)
    lines.append(f"Distil-Whisper ({MODEL_NAME}) on LibriSpeech test-clean — Step 2: Voice Note Transcription")
    lines.append("=" * 84)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 84)
    lines.append(f"WER (Word Error Rate)  : {overall_wer:.4f}   ({overall_wer*100:.2f}% of words wrong)")
    lines.append(f"CER (Character Error Rate): {overall_cer:.4f}   ({overall_cer*100:.2f}% of characters wrong)")
    lines.append(f"Perfect transcriptions : {perfect_clips}/{len(results)}   ({perfect_clips/len(results)*100:.1f}% exact match after normalization)")
    lines.append(f"Avg Time               : {avg_time_s*1000:.0f} ms per clip")
    lines.append(f"Clips evaluated        : {len(results)}")
    lines.append("-" * 84)
    lines.append("")
    lines.append("WORST 10 CLIPS BY WER (for qualitative error inspection)")
    lines.append("-" * 84)
    worst = sorted(zip(results, per_clip_wer), key=lambda x: -x[1])[:10]
    for r, w in worst:
        lines.append(f"\n{r['utt_id']}  (WER={w:.2f})")
        lines.append(f"  REF: {r['reference_norm']}")
        lines.append(f"  HYP: {r['hypothesis_norm']}")
    lines.append("")
    lines.append("=" * 84)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    return {
        "wer": overall_wer,
        "cer": overall_cer,
        "perfect_rate": perfect_clips / len(results),
        "avg_time_ms": avg_time_s * 1000,
        "n_clips": len(results),
        "per_clip_wer": per_clip_wer,
        "report_text": report_text,
    }


# Writes the result charts used in the report.
def save_graphs(metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    labels = ["WER", "CER", "Perfect\nMatch Rate"]
    values = [metrics["wer"], metrics["cer"], metrics["perfect_rate"]]
    bars = plt.bar(labels, values, color="#7C3AED")
    plt.ylim(0, max(1.0, max(values) * 1.2))
    plt.title(f"Distil-Whisper on LibriSpeech test-clean — Overall Summary\n"
              f"(Avg. time per clip: {metrics['avg_time_ms']:.0f} ms, n={metrics['n_clips']})")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "overall_summary.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(metrics["per_clip_wer"], bins=30, color="#2563EB", edgecolor="white")
    plt.xlabel("Per-clip WER")
    plt.ylabel("Number of clips")
    plt.title("Distil-Whisper on LibriSpeech test-clean — Per-Clip WER Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "wer_distribution.png"), dpi=150)
    plt.close()

    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(metrics["report_text"] + "\n")

    print(f"\nGraphs and metrics.txt saved to: {results_dir}")


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Evaluate Distil-Whisper on LibriSpeech test-clean")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    results = run_evaluation(args.dataset_dir, args.sample)
    metrics = report_metrics(results)
    save_graphs(metrics, RESULTS_DIR)


if __name__ == "__main__":
    main()
