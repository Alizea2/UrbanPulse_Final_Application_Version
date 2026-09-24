"""Per-stage and end-to-end latency testing for the UrbanPulse backend.
Runs against the live Flask API with models already warm-loaded.
"""
import argparse
import time
import statistics
import requests

API_BASE = "http://127.0.0.1:5001"
N_RUNS = 20
# Warm-up runs are timed but excluded from the stats.
N_WARMUP = 2
AUDIO_PATH = "test.wav"
IMAGE_PATH = "Test/Model Test/Datasets/Scene Classification/places365_torch_uncertainty/archive/Places365_val_00004488.jpg"


def percentile(values, pct):
    """Linear-interpolated percentile."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def time_calls(fn, n=N_RUNS, warmup=N_WARMUP):
    times = []
    result = None
    for i in range(warmup):
        start = time.perf_counter()
        result = fn()
        print(f"    warm-up {i+1}/{warmup}: {time.perf_counter() - start:.3f}s (excluded)")
    for i in range(n):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    run {i+1}/{n}: {elapsed:.3f}s")
    return times, result


def summarize(name, times):
    mean = statistics.mean(times)
    p95 = percentile(times, 95)
    print(f"\n{name}: mean={mean:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s  p95={p95:.3f}s\n")
    return {"stage": name, "mean": mean, "min": min(times), "max": max(times), "p95": p95, "n": len(times)}


def call_analyze():
    with open(AUDIO_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/analyze", files={"audio": f})
    r.raise_for_status()
    return r.json()


def call_scene():
    with open(IMAGE_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/scene", files={"image": f})
    r.raise_for_status()
    return r.json()


def call_transcribe():
    with open(AUDIO_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/transcribe", files={"audio": f})
    r.raise_for_status()
    return r.json()


def call_wellbeing(sound_label, scene_label, transcript, emotion):
    r = requests.post(f"{API_BASE}/api/wellbeing", json={
        "sound_label": sound_label,
        "scene_label": scene_label,
        "transcript": transcript,
        "emotion": emotion,
    })
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(description="UrbanPulse backend latency benchmark")
    parser.add_argument(
        "--steps13-only", action="store_true",
        help="Measure only Steps 1-3; skips the two stages that call the Ollama LLM.")
    args = parser.parse_args()

    results = []

    print(f"=== Sound Classification (EfficientAT) — /api/analyze, n={N_RUNS} ===")
    times, last = time_calls(call_analyze)
    results.append(summarize("Sound Classification (EfficientAT)", times))
    sound_label = last["results"][0]["label"]

    print(f"=== Scene Classification (CLIP) — /api/scene, n={N_RUNS} ===")
    times, last = time_calls(call_scene)
    results.append(summarize("Scene Classification (CLIP)", times))
    scene_label = last["scene"]

    print(f"=== Transcription + Emotion (wav2vec2 + SER) — /api/transcribe, n={N_RUNS} ===")
    times, last = time_calls(call_transcribe)
    results.append(summarize("Transcription + Emotion (wav2vec2 x2)", times))
    # test.wav has no speech, so transcription returns "" -- /api/wellbeing
    # rejects an empty transcript, so substitute a placeholder for the
    # wellbeing/end-to-end latency stages (we're timing the LLM call here,
    # not testing transcription content).
    transcript = last["text"] or "It's pretty loud outside but I feel okay."
    emotion = last["emotion"]

    if not args.steps13_only:
        print(f"=== Wellbeing Scoring (Qwen2.5) — /api/wellbeing, n={N_RUNS} ===")
        times, last = time_calls(lambda: call_wellbeing(sound_label, scene_label, transcript, emotion))
        results.append(summarize("Wellbeing Scoring (Qwen2.5)", times))

    # Steps 1-3 timed as one path -- the scope of the Table 3.2.2 requirement.
    print(f"=== Steps 1-3 only (analyze -> scene -> transcribe), n={N_RUNS} ===")
    steps13_times = []
    for i in range(N_WARMUP):
        call_analyze(); call_scene(); call_transcribe()
    for i in range(N_RUNS):
        start = time.perf_counter()
        call_analyze()
        call_scene()
        call_transcribe()
        elapsed = time.perf_counter() - start
        steps13_times.append(elapsed)
        print(f"    run {i+1}/{N_RUNS}: {elapsed:.3f}s")
    s13 = summarize("Steps 1-3 (paired, requirement scope)", steps13_times)
    results.append(s13)
    verdict = "MEETS" if s13["p95"] <= 6.0 else "EXCEEDS"
    print(f"    -> p95 = {s13['p95']:.3f}s vs the 6s budget in Table 3.2.2: {verdict}\n")

    if not args.steps13_only:
        print("=== End-to-end pipeline (sequential: analyze -> scene -> transcribe -> wellbeing) ===")
        e2e_times = []
        for i in range(N_RUNS):
            start = time.perf_counter()
            a = call_analyze()
            s = call_scene()
            t = call_transcribe()
            call_wellbeing(a["results"][0]["label"], s["scene"], t["text"] or "It's pretty loud outside but I feel okay.", t["emotion"])
            elapsed = time.perf_counter() - start
            e2e_times.append(elapsed)
            print(f"    run {i+1}/{N_RUNS}: {elapsed:.3f}s")
        results.append(summarize("End-to-end pipeline (sequential)", e2e_times))
    else:
        print("(Skipped: Wellbeing Scoring and End-to-end pipeline — both call the Ollama LLM.)\n")

    print("\n\n===== SUMMARY =====")
    print(f"{'Stage':<45} {'Mean (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'P95 (s)':>10}")
    for r in results:
        print(f"{r['stage']:<45} {r['mean']:>10.3f} {r['min']:>10.3f} {r['max']:>10.3f} {r['p95']:>10.3f}")


if __name__ == "__main__":
    main()
