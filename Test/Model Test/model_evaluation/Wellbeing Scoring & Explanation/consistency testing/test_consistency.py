"""Property-based consistency evaluation for the Step 4 wellbeing scorer.
Evaluation only -- not part of the app.

There is no published ground truth for this task, so instead of grading
against invented absolute scores, this checks invariants that must hold
whatever the "true" score is:

1-4. Ranking monotonicity -- vary one signal at a time (sound, emotion,
     scene, transcript) across a worst-to-best ordering with the others
     held neutral, and measure Spearman rho. Scene and transcript are
     bounded modifiers, so their spread is reported alongside rho.
5.   Test-retest reliability -- identical inputs run N times; a
     trustworthy scorer should be near-deterministic.
6.   Extremes sanity check -- all-worst and all-best inputs should land
     in the correct half of the scale.

Needs Ollama running locally with the target model pulled:

    python3 "<this file>" --retest-n 5 --retest-repeats 8
"""

import argparse
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Wellbeing Scoring & Explanation/consistency testing
TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
PROJECT_ROOT = os.path.dirname(TESTS_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import models.wellbeing_scorer as wellbeing_scorer
from models.wellbeing_scorer import (
    PLACE_PROMPT_WITH_SCENE, PERSON_PROMPT_TEMPLATE,
    SOUND_SCORE_RUBRIC, EMOTION_SCORE_RUBRIC, SCENE_SCORE_RUBRIC, TRANSCRIPT_SCORE_RUBRIC,
    _call_ollama, _extract_json, combine_scores,
)

# Anchors near 0.50 in their own baseline tables, used to hold three signals
# still while the fourth is varied, so they add no net pull.
NEUTRAL_SOUND = "large-crowd"  # baseline 0.40, closest SONYC-UST class to neutral
NEUTRAL_EMOTION = "Calm"
NEUTRAL_SCENE = "inside a building"  # baseline 0.50 — deliberately neutral (see dataset builder)
NEUTRAL_TRANSCRIPT = "It's fine here, nothing special either way."  # baseline 0.50

# Sound sequence, worst -> best, grounded in published psychoacoustic/
# soundscape research (WHO noise annoyance guidelines, Axelsson's
# soundscape circumplex model).
SOUND_RANK_SEQUENCE = [
    ("jackhammer", 0.08),
    ("siren", 0.10),
    ("car-alarm", 0.15),
    ("car-horn", 0.20),
    ("person-or-small-group-shouting", 0.25),
    ("large-sounding-engine", 0.28),
    ("large-crowd", 0.40),
    ("mobile-music", 0.55),
    ("stationary-music", 0.60),
    ("person-or-small-group-talking", 0.65),
]

# Emotion sequence, worst -> best, per Russell's (1980) circumplex
# valence ordering (also matches EMOTION_BASELINES in the dataset builder).
EMOTION_RANK_SEQUENCE = [
    ("Stressed", "I'm so behind on everything today, this is a lot."),
    ("Annoyed", "Honestly this is getting really irritating now."),
    ("Anxious", "I don't know, I'm feeling a bit on edge right now."),
    ("Calm", "It's a quiet afternoon, feels nice out here."),
    ("Content", "Had a great coffee and I'm feeling good today."),
]

# Scene sequence, worst -> best, matching SCENE_CATEGORY_BASELINES
# (Ulrich, 1984; Kaplan & Kaplan, 1989).
SCENE_RANK_SEQUENCE = [
    ("a construction site", 0.08),
    ("an industrial area", 0.12),
    ("a traffic intersection", 0.20),
    ("a busy street", 0.25),
    ("a train station or airport", 0.30),
    ("a market or bazaar", 0.40),
    ("an office", 0.48),
    ("a café or restaurant", 0.55),
    ("a residential street", 0.65),
    ("a forest or wooded area", 0.95),
]

# Transcript sentiment sequence, worst -> best. Emotion is held neutral
# throughout, so this measures the wording alone rather than an echo of it.
TRANSCRIPT_RANK_SEQUENCE = [
    ("I really can't stand this anymore, it's awful here.", 0.08),
    ("I'm getting pretty fed up, this is a lot to deal with.", 0.20),
    ("This is starting to get to me a little.", 0.30),
    ("It's fine here, nothing special either way.", 0.50),
    ("Things are pretty good right now, can't complain.", 0.70),
    ("This is such a nice spot, I'm really enjoying being here.", 0.80),
    ("I am absolutely loving this moment, everything feels perfect.", 0.95),
]

# 20 diverse inputs for the test-retest check, spanning the full range.
RETEST_INPUTS = [
    ("jackhammer", "Stressed", "I'm so behind on everything today, this is a lot.", "a construction site"),
    ("siren", "Anxious", "My heart's racing a little, not sure what's going on.", "a traffic intersection"),
    ("car-alarm", "Annoyed", "Can this just stop already, I've had enough.", "a parking lot"),
    ("large-sounding-engine", "Calm", "It's a quiet afternoon, feels nice out here.", "a residential street"),
    ("person-or-small-group-talking", "Content", "Had a great coffee and I'm feeling good today.", "a café or restaurant"),
    ("stationary-music", "Calm", "This spot is really peaceful right now.", "a park"),
    ("large-crowd", "Anxious", "Something about this street is making me uneasy.", "a busy street"),
    ("dog-barking-whining", "Annoyed", "This is so annoying, I just want some quiet.", "a quiet alley"),
    ("rock-drill", "Stressed", "I really need this to be over, I can't focus.", "an industrial area"),
    ("mobile-music", "Content", "Just relaxing and enjoying the day so far.", "a garden"),
    ("car-horn", "Stressed", "Everything is piling up and I just need a minute.", "a highway or motorway"),
    ("amplified-speech", "Calm", "Taking a slow walk, nothing much going on.", "a school or university campus"),
    ("ice-cream-truck", "Content", "This is honestly a really nice moment right now.", "a playground"),
    ("chainsaw", "Anxious", "I keep looking around, not sure why I'm nervous.", "a construction site"),
    ("reverse-beeper", "Annoyed", "I'm getting pretty fed up with this whole thing.", "a parking lot"),
    ("non-machinery-impact", "Calm", "I'm just sitting outside enjoying some fresh air.", "a beach or waterfront"),
    ("pile-driver", "Stressed", "I'm running late and nothing is going right.", "an industrial area"),
    ("person-or-small-group-shouting", "Anxious", "I don't know, I'm feeling a bit on edge right now.", "a market or bazaar"),
    ("small-sounding-engine", "Content", "Everything's going well, I'm in a good mood.", "a home interior"),
    ("hoe-ram", "Annoyed", "Honestly this is getting really irritating now.", "a shopping mall"),
]

# Obviously-worst and obviously-best combinations across ALL FOUR
# signals at once, for the coarse extremes sanity check. No precise
# value expected, just the correct half of the [0, 1] scale.
WORST_CASES = [
    ("jackhammer", "Stressed", "I really can't stand this anymore, it's awful here.", "a construction site"),
    ("pile-driver", "Stressed", "I'm getting pretty fed up, this is a lot to deal with.", "an industrial area"),
    ("siren", "Annoyed", "Can this just stop already, I've had enough.", "a highway or motorway"),
]
BEST_CASES = [
    ("person-or-small-group-talking", "Content", "I am absolutely loving this moment, everything feels perfect.", "a forest or wooded area"),
    ("stationary-music", "Calm", "This is such a nice spot, I'm really enjoying being here.", "a park"),
    ("mobile-music", "Content", "Things are pretty good right now, can't complain.", "a garden"),
]
WORST_CASE_CEILING = 0.35
BEST_CASE_FLOOR = 0.65


def get_wellbeing_score(sound_label, emotion, transcript, scene_label, temperature=0.0):
    """Runs only the two sub-score calls (Place: sound+scene, Person:
    emotion+transcript) + the deterministic combine — skips the
    explanation call, which is irrelevant for these consistency checks
    and roughly halves the number of LLM calls needed. Mirrors
    models.wellbeing_scorer.score_wellbeing()'s first two stages
    exactly."""
    place_prompt = PLACE_PROMPT_WITH_SCENE.format(
        sound_label=sound_label, scene_label=scene_label,
        sound_rubric=SOUND_SCORE_RUBRIC, scene_rubric=SCENE_SCORE_RUBRIC,
    )
    raw_place = _call_ollama(place_prompt, temperature=temperature)
    parsed_place = _extract_json(raw_place)
    sound_score = max(0.0, min(1.0, float(parsed_place["sound_score"])))
    scene_score = max(0.0, min(1.0, float(parsed_place["scene_score"])))

    person_prompt = PERSON_PROMPT_TEMPLATE.format(
        emotion=emotion, transcript=transcript,
        emotion_rubric=EMOTION_SCORE_RUBRIC, transcript_rubric=TRANSCRIPT_SCORE_RUBRIC,
    )
    raw_person = _call_ollama(person_prompt, temperature=temperature)
    parsed_person = _extract_json(raw_person)
    emotion_score = max(0.0, min(1.0, float(parsed_person["emotion_score"])))
    transcript_score = max(0.0, min(1.0, float(parsed_person["transcript_score"])))

    final_score = combine_scores(sound_score, emotion_score, transcript_score, scene_score)
    return final_score, sound_score, emotion_score, transcript_score, scene_score


# Invariant 1: does the score follow the sound ordering?
def run_sound_ranking_test():
    print("Running sound-ranking monotonicity test...")
    scores = []
    for sound_label, _ in SOUND_RANK_SEQUENCE:
        score, *_ = get_wellbeing_score(sound_label, NEUTRAL_EMOTION, NEUTRAL_TRANSCRIPT, NEUTRAL_SCENE)
        scores.append(score)
        print(f"  {sound_label:<35} -> {score:.2f}")

    literature_ranks = list(range(len(SOUND_RANK_SEQUENCE)))
    rho, p_value = spearmanr(literature_ranks, scores)
    return {
        "sequence": [s for s, _ in SOUND_RANK_SEQUENCE],
        "scores": scores,
        "spearman_rho": rho,
        "p_value": p_value,
    }


# Invariant 2: does the score follow the emotion ordering?
def run_emotion_ranking_test():
    print("Running emotion-ranking monotonicity test...")
    scores = []
    for emotion, transcript in EMOTION_RANK_SEQUENCE:
        score, *_ = get_wellbeing_score(NEUTRAL_SOUND, emotion, transcript, NEUTRAL_SCENE)
        scores.append(score)
        print(f"  {emotion:<12} -> {score:.2f}")

    literature_ranks = list(range(len(EMOTION_RANK_SEQUENCE)))
    rho, p_value = spearmanr(literature_ranks, scores)
    return {
        "sequence": [e for e, _ in EMOTION_RANK_SEQUENCE],
        "scores": scores,
        "spearman_rho": rho,
        "p_value": p_value,
    }


# Invariant 3: does the score follow the scene ordering?
def run_scene_ranking_test():
    print("Running scene-ranking monotonicity test...")
    scores = []
    for scene_label, _ in SCENE_RANK_SEQUENCE:
        score, *_ = get_wellbeing_score(NEUTRAL_SOUND, NEUTRAL_EMOTION, NEUTRAL_TRANSCRIPT, scene_label)
        scores.append(score)
        print(f"  {scene_label:<28} -> {score:.2f}")

    literature_ranks = list(range(len(SCENE_RANK_SEQUENCE)))
    rho, p_value = spearmanr(literature_ranks, scores)
    return {
        "sequence": [s for s, _ in SCENE_RANK_SEQUENCE],
        "scores": scores,
        "spearman_rho": rho,
        "p_value": p_value,
        "spread": max(scores) - min(scores),
    }


# Invariant 4: does the score follow the transcript ordering?
def run_transcript_ranking_test():
    print("Running transcript-ranking monotonicity test...")
    scores = []
    for transcript, _ in TRANSCRIPT_RANK_SEQUENCE:
        score, *_ = get_wellbeing_score(NEUTRAL_SOUND, NEUTRAL_EMOTION, transcript, NEUTRAL_SCENE)
        scores.append(score)
        print(f"  {transcript[:40]:<42} -> {score:.2f}")

    literature_ranks = list(range(len(TRANSCRIPT_RANK_SEQUENCE)))
    rho, p_value = spearmanr(literature_ranks, scores)
    return {
        "sequence": [t for t, _ in TRANSCRIPT_RANK_SEQUENCE],
        "scores": scores,
        "spearman_rho": rho,
        "p_value": p_value,
        "spread": max(scores) - min(scores),
    }


# Invariant 5: identical inputs should give the same score each time.
def run_retest_reliability_test(n_inputs, n_repeats):
    print(f"Running test-retest reliability ({n_inputs} inputs x {n_repeats} repeats)...")
    inputs = RETEST_INPUTS[:n_inputs]
    per_input_stdevs = []
    per_input_results = []

    for sound_label, emotion, transcript, scene_label in inputs:
        run_scores = []
        for _ in range(n_repeats):
            score, *_ = get_wellbeing_score(sound_label, emotion, transcript, scene_label)
            run_scores.append(score)
        stdev = statistics.pstdev(run_scores)
        per_input_stdevs.append(stdev)
        per_input_results.append({
            "sound_label": sound_label, "emotion": emotion, "scene_label": scene_label,
            "scores": run_scores, "stdev": stdev,
        })
        print(f"  {sound_label:<30} + {emotion:<10} + {scene_label:<24} -> scores={run_scores}  stdev={stdev:.4f}")

    return {
        "per_input": per_input_results,
        "mean_stdev": statistics.mean(per_input_stdevs),
        "max_stdev": max(per_input_stdevs),
    }


# Invariant 6: all-worst and all-best must land in the right half.
def run_extremes_test():
    print("Running extremes sanity check...")
    worst_results = []
    for sound_label, emotion, transcript, scene_label in WORST_CASES:
        score, *_ = get_wellbeing_score(sound_label, emotion, transcript, scene_label)
        passed = score < WORST_CASE_CEILING
        worst_results.append({"sound_label": sound_label, "emotion": emotion, "scene_label": scene_label,
                               "score": score, "passed": passed})
        print(f"  WORST  {sound_label:<30} + {emotion:<10} + {scene_label:<24} -> {score:.2f} "
              f"({'PASS' if passed else 'FAIL'})")

    best_results = []
    for sound_label, emotion, transcript, scene_label in BEST_CASES:
        score, *_ = get_wellbeing_score(sound_label, emotion, transcript, scene_label)
        passed = score > BEST_CASE_FLOOR
        best_results.append({"sound_label": sound_label, "emotion": emotion, "scene_label": scene_label,
                              "score": score, "passed": passed})
        print(f"  BEST   {sound_label:<30} + {emotion:<10} + {scene_label:<24} -> {score:.2f} "
              f"({'PASS' if passed else 'FAIL'})")

    all_results = worst_results + best_results
    pass_rate = sum(1 for r in all_results if r["passed"]) / len(all_results)
    return {"worst": worst_results, "best": best_results, "pass_rate": pass_rate}


# Computes the headline metrics and writes metrics.txt.
def report_metrics(model_name, sound_test, emotion_test, scene_test, transcript_test, retest_test, extremes_test, elapsed_s):
    lines = []
    lines.append("=" * 90)
    lines.append(f"Property-Based Consistency Evaluation — {model_name} — Step 4: Wellbeing Scoring (4-input)")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Why this evaluation exists: no validated ground truth exists for this fused task, so")
    lines.append("instead of grading against a possibly-wrong invented absolute number, this checks")
    lines.append("invariants that must hold regardless of what the 'true' score even is.")
    lines.append("")
    lines.append("-" * 90)
    lines.append("1. SOUND-RANKING MONOTONICITY")
    lines.append("-" * 90)
    lines.append(f"Literature-ranked worst -> best sequence (fixed emotion={NEUTRAL_EMOTION}, scene={NEUTRAL_SCENE}):")
    for (sound, _), score in zip(SOUND_RANK_SEQUENCE, sound_test["scores"]):
        lines.append(f"  {sound:<35} -> {score:.2f}")
    lines.append(f"Spearman rho: {sound_test['spearman_rho']:.4f}  (p={sound_test['p_value']:.4f})")
    lines.append("(+1.0 = perfect agreement with literature ranking, 0 = no relationship, -1.0 = backwards)")
    lines.append("")
    lines.append("-" * 90)
    lines.append("2. EMOTION-RANKING MONOTONICITY")
    lines.append("-" * 90)
    lines.append(f"Literature-ranked worst -> best sequence (fixed sound={NEUTRAL_SOUND}, scene={NEUTRAL_SCENE}):")
    for (emotion, _), score in zip(EMOTION_RANK_SEQUENCE, emotion_test["scores"]):
        lines.append(f"  {emotion:<12} -> {score:.2f}")
    lines.append(f"Spearman rho: {emotion_test['spearman_rho']:.4f}  (p={emotion_test['p_value']:.4f})")
    lines.append("")
    lines.append("-" * 90)
    lines.append("3. SCENE-RANKING MONOTONICITY")
    lines.append("-" * 90)
    lines.append(f"Literature-ranked worst -> best sequence (fixed sound={NEUTRAL_SOUND}, emotion={NEUTRAL_EMOTION}):")
    for (scene, _), score in zip(SCENE_RANK_SEQUENCE, scene_test["scores"]):
        lines.append(f"  {scene:<28} -> {score:.2f}")
    lines.append(f"Spearman rho: {scene_test['spearman_rho']:.4f}  (p={scene_test['p_value']:.4f})")
    lines.append(f"Score spread across sequence: {scene_test['spread']:.4f} "
                  f"(expected small — scene is a bounded +/-0.10 modifier, not a core 50%-weighted signal)")
    lines.append("")
    lines.append("-" * 90)
    lines.append("4. TRANSCRIPT-RANKING MONOTONICITY")
    lines.append("-" * 90)
    lines.append(f"Sentiment-ranked worst -> best sequence (fixed sound={NEUTRAL_SOUND}, emotion={NEUTRAL_EMOTION}):")
    for (transcript, _), score in zip(TRANSCRIPT_RANK_SEQUENCE, transcript_test["scores"]):
        lines.append(f"  {transcript[:50]:<52} -> {score:.2f}")
    lines.append(f"Spearman rho: {transcript_test['spearman_rho']:.4f}  (p={transcript_test['p_value']:.4f})")
    lines.append(f"Score spread across sequence: {transcript_test['spread']:.4f} "
                  f"(expected small — transcript is a bounded +/-0.10 modifier, not a core 50%-weighted signal)")
    lines.append("")
    lines.append("-" * 90)
    lines.append("5. TEST-RETEST RELIABILITY")
    lines.append("-" * 90)
    for r in retest_test["per_input"]:
        lines.append(f"  {r['sound_label']:<30} + {r['emotion']:<10} + {r['scene_label']:<24} "
                      f"scores={r['scores']}  stdev={r['stdev']:.4f}")
    lines.append(f"Mean stdev across all inputs: {retest_test['mean_stdev']:.4f}")
    lines.append(f"Max stdev (worst single input): {retest_test['max_stdev']:.4f}")
    lines.append("(Lower is better — near 0.0 means the model gives the same score every time)")
    lines.append("")
    lines.append("-" * 90)
    lines.append("6. EXTREMES SANITY CHECK")
    lines.append("-" * 90)
    for r in extremes_test["worst"]:
        lines.append(f"  WORST  {r['sound_label']:<30} + {r['emotion']:<10} + {r['scene_label']:<24} -> {r['score']:.2f} "
                      f"(expected < {WORST_CASE_CEILING}) [{'PASS' if r['passed'] else 'FAIL'}]")
    for r in extremes_test["best"]:
        lines.append(f"  BEST   {r['sound_label']:<30} + {r['emotion']:<10} + {r['scene_label']:<24} -> {r['score']:.2f} "
                      f"(expected > {BEST_CASE_FLOOR}) [{'PASS' if r['passed'] else 'FAIL'}]")
    lines.append(f"Pass rate: {extremes_test['pass_rate']*100:.1f}%")
    lines.append("")
    lines.append("-" * 90)
    lines.append("SUMMARY")
    lines.append("-" * 90)
    lines.append(f"Sound-ranking Spearman rho      : {sound_test['spearman_rho']:.4f}")
    lines.append(f"Emotion-ranking Spearman rho    : {emotion_test['spearman_rho']:.4f}")
    lines.append(f"Scene-ranking Spearman rho      : {scene_test['spearman_rho']:.4f}")
    lines.append(f"Transcript-ranking Spearman rho : {transcript_test['spearman_rho']:.4f}")
    lines.append(f"Test-retest mean stdev          : {retest_test['mean_stdev']:.4f}")
    lines.append(f"Extremes pass rate              : {extremes_test['pass_rate']*100:.1f}%")
    lines.append(f"Total evaluation time            : {elapsed_s:.0f}s")
    lines.append("=" * 90)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    return report_text


# Writes the result charts used in the report.
def save_graphs(sound_test, emotion_test, scene_test, transcript_test, retest_test, extremes_test, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    def plot_ranking(test, title, filename, color, xtick_fontsize=8):
        plt.figure(figsize=(9, 6))
        x = list(range(len(test["sequence"])))
        plt.plot(x, test["scores"], marker="o", color=color, linewidth=2)
        labels = [s if len(s) <= 22 else s[:20] + "…" for s in test["sequence"]]
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=xtick_fontsize)
        plt.ylabel("Model wellbeing score")
        plt.ylim(0, 1.05)
        plt.title(f"{title} (Spearman rho={test['spearman_rho']:.3f})\nLiterature order: worst -> best (left to right)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename), dpi=150)
        plt.close()

    plot_ranking(sound_test, "Sound-Ranking Monotonicity", "sound_ranking_monotonicity.png", "#0F766E")
    plot_ranking(emotion_test, "Emotion-Ranking Monotonicity", "emotion_ranking_monotonicity.png", "#7C3AED", xtick_fontsize=9)
    plot_ranking(scene_test, "Scene-Ranking Monotonicity", "scene_ranking_monotonicity.png", "#0369A1")
    plot_ranking(transcript_test, "Transcript-Ranking Monotonicity", "transcript_ranking_monotonicity.png", "#B45309", xtick_fontsize=6)

    # Test-retest reliability: stdev per input
    plt.figure(figsize=(11, 6))
    labels = [f"{r['sound_label'][:12]}+{r['emotion'][:4]}+{r['scene_label'][:10]}" for r in retest_test["per_input"]]
    stdevs = [r["stdev"] for r in retest_test["per_input"]]
    plt.bar(range(len(labels)), stdevs, color="#059669")
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Score standard deviation across repeats")
    plt.title(f"Test-Retest Reliability (mean stdev={retest_test['mean_stdev']:.4f})\nLower = more consistent")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_retest_reliability.png"), dpi=150)
    plt.close()

    # Extremes pass/fail
    plt.figure(figsize=(8, 6))
    all_cases = extremes_test["worst"] + extremes_test["best"]
    labels = [f"{r['sound_label'][:12]}+{r['emotion'][:4]}+{r['scene_label'][:10]}" for r in all_cases]
    scores = [r["score"] for r in all_cases]
    colors = ["#059669" if r["passed"] else "#DC2626" for r in all_cases]
    plt.bar(range(len(labels)), scores, color=colors)
    plt.axhline(WORST_CASE_CEILING, color="gray", linestyle="--", linewidth=1, label=f"worst-case ceiling ({WORST_CASE_CEILING})")
    plt.axhline(BEST_CASE_FLOOR, color="gray", linestyle=":", linewidth=1, label=f"best-case floor ({BEST_CASE_FLOOR})")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Model wellbeing score")
    plt.title(f"Extremes Sanity Check (pass rate={extremes_test['pass_rate']*100:.0f}%)\nGreen=pass, Red=fail")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "extremes_sanity_check.png"), dpi=150)
    plt.close()

    print(f"\nGraphs and metrics.txt saved to: {results_dir}")


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Property-based consistency evaluation for Step 4 (4-input)")
    parser.add_argument("--model", default=wellbeing_scorer.MODEL_NAME,
                         help="Ollama model tag to test (must already be pulled)")
    parser.add_argument("--retest-n", type=int, default=20, help="Number of inputs for the test-retest check")
    parser.add_argument("--retest-repeats", type=int, default=5, help="Repeats per input for the test-retest check")
    args = parser.parse_args()

    # wellbeing_scorer._call_ollama() reads the module-level MODEL_NAME at
    # call time, so overriding it here redirects every call in this run
    # to the requested model without needing to duplicate the Ollama
    # request logic.
    wellbeing_scorer.MODEL_NAME = args.model
    print(f"Testing model: {args.model}")

    results_dir = os.path.join(TESTS_ROOT, "Test-Results", "Wellbeing Scoring & Explanation", "consistency testing",
                                f"{args.model.replace(':', '_')}_Consistency")

    t_start = time.time()

    sound_test = run_sound_ranking_test()
    emotion_test = run_emotion_ranking_test()
    scene_test = run_scene_ranking_test()
    transcript_test = run_transcript_ranking_test()
    retest_test = run_retest_reliability_test(args.retest_n, args.retest_repeats)
    extremes_test = run_extremes_test()

    elapsed_s = time.time() - t_start

    report_text = report_metrics(args.model, sound_test, emotion_test, scene_test, transcript_test,
                                  retest_test, extremes_test, elapsed_s)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(report_text + "\n")

    save_graphs(sound_test, emotion_test, scene_test, transcript_test, retest_test, extremes_test, results_dir)


if __name__ == "__main__":
    main()
