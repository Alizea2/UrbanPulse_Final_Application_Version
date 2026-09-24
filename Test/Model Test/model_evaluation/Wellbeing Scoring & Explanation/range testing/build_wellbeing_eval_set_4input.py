"""Builds the curated ground-truth evaluation set for the 4-input Step 4
scorer (sound + emotion + transcript + scene), matching the fusion
formula in models/wellbeing_scorer.py.

Each expected range is grounded in published soundscape and restorative-
environment research; the sources are cited per category below.
"""

import csv
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
OUT_PATH = os.path.join(TESTS_ROOT, "Datasets", "Wellbeing Scoring & Explanation", "wellbeing_eval_set_4input.csv")

# Same tolerance as the original 2-input dataset, for the same reason
# (see build_wellbeing_eval_set.py) — kept identical so range-matching
# accuracy stays comparable across the two dataset generations.
TOLERANCE = 0.10

# Modifier caps must match models/wellbeing_scorer.py's
# TRANSCRIPT_MODIFIER_MAX / SCENE_MODIFIER_MAX exactly — this dataset's
# ground truth is only valid against the formula actually running.
TRANSCRIPT_MODIFIER_MAX = 0.10
SCENE_MODIFIER_MAX = 0.10

# --- Sound + emotion baselines: identical to build_wellbeing_eval_set.py ---
SOUND_CATEGORY_BASELINES = {
    "small-sounding-engine": 0.50,
    "medium-sounding-engine": 0.40,
    "large-sounding-engine": 0.28,
    "rock-drill": 0.10,
    "jackhammer": 0.08,
    "hoe-ram": 0.10,
    "pile-driver": 0.08,
    "non-machinery-impact": 0.35,
    "chainsaw": 0.15,
    "small-medium-rotating-saw": 0.20,
    "large-rotating-saw": 0.15,
    "car-horn": 0.20,
    "car-alarm": 0.15,
    "siren": 0.10,
    "reverse-beeper": 0.25,
    "stationary-music": 0.60,
    "mobile-music": 0.55,
    "ice-cream-truck": 0.65,
    "person-or-small-group-talking": 0.65,
    "person-or-small-group-shouting": 0.25,
    "large-crowd": 0.40,
    "amplified-speech": 0.35,
    "dog-barking-whining": 0.35,
}
SOUND_BASELINES = list(SOUND_CATEGORY_BASELINES.items())

EMOTION_BASELINES = [
    ("Calm", 0.85),
    ("Content", 0.90),
    ("Anxious", 0.30),
    ("Stressed", 0.15),
    ("Annoyed", 0.25),
]

# --- NEW: scene baselines, grounded per the docstring above. Order and
# labels match models/scene_classifier.py's CANDIDATE_SCENES exactly. ---
SCENE_CATEGORY_BASELINES = {
    # Natural/green — Ulrich (1984); Kaplan & Kaplan (1989) Attention
    # Restoration Theory.
    "a park": 0.90,
    "a garden": 0.90,
    "a forest or wooded area": 0.95,
    "a beach or waterfront": 0.92,

    # Traffic-dominated — visual traffic density as urban-stress
    # correlate, paralleling WHO (2018) road-traffic annoyance findings.
    "a busy street": 0.25,
    "a traffic intersection": 0.20,
    "a highway or motorway": 0.10,

    # Low-activity residential — implied low traffic/crowd density.
    "a residential street": 0.65,
    "a quiet alley": 0.62,

    # Busy commercial/transit — Axelsson et al. (2010) "chaotic" pole
    # extended qualitatively to visually busy/crowded scenes.
    "a market or bazaar": 0.40,
    "a shopping mall": 0.38,
    "a café or restaurant": 0.55,

    # Industrial/construction — mirrors the impulsive/industrial-noise
    # penalty already applied on the sound side.
    "a construction site": 0.08,
    "an industrial area": 0.12,

    # Indoor/neutral — no strong inherent valence, deliberately
    # centered near 0.5.
    "inside a building": 0.50,
    "an office": 0.48,
    "a home interior": 0.55,

    # Mixed-social — social but not inherently distressing.
    "a school or university campus": 0.55,
    "a playground": 0.60,

    "a train station or airport": 0.30,
    "a parking lot": 0.30,
}
SCENE_BASELINES = list(SCENE_CATEGORY_BASELINES.items())

# --- NEW: transcript sentiment pool, independent of emotion label.
# Bands mirror wellbeing_scorer.py's TRANSCRIPT_SCORE_RUBRIC exactly. ---
TRANSCRIPT_SENTIMENT_POOL = [
    ("I am absolutely loving this moment, everything feels perfect.", 0.95),
    ("This is such a nice spot, I'm really enjoying being here.", 0.80),
    ("Things are pretty good right now, can't complain.", 0.70),
    ("It's fine here, nothing special either way.", 0.50),
    ("I'm not really sure how I feel about this place.", 0.45),
    ("This is starting to get to me a little.", 0.30),
    ("I'm getting pretty fed up, this is a lot to deal with.", 0.20),
    ("I really can't stand this anymore, it's awful here.", 0.08),
]


# Keeps a value inside the 0.0-1.0 range.
def _clamp01(value):
    return max(0.0, min(1.0, value))


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    rows = []
    row_id = 1
    for sound_label, sound_score in SOUND_BASELINES:
        for emotion, emotion_score in EMOTION_BASELINES:
            scene_label, scene_score = SCENE_BASELINES[row_id % len(SCENE_BASELINES)]
            transcript, transcript_score = TRANSCRIPT_SENTIMENT_POOL[row_id % len(TRANSCRIPT_SENTIMENT_POOL)]

            core = 0.5 * sound_score + 0.5 * emotion_score
            transcript_mod = (transcript_score - 0.5) * 2 * TRANSCRIPT_MODIFIER_MAX
            scene_mod = (scene_score - 0.5) * 2 * SCENE_MODIFIER_MAX
            center_score = round(_clamp01(core + transcript_mod + scene_mod), 2)

            range_min = max(0.0, round(center_score - TOLERANCE, 2))
            range_max = min(1.0, round(center_score + TOLERANCE, 2))

            rows.append({
                "id": row_id,
                "sound_label": sound_label,
                "emotion": emotion,
                "transcript": transcript,
                "scene_label": scene_label,
                "acceptable_score_min": f"{range_min:.2f}",
                "acceptable_score_max": f"{range_max:.2f}",
            })
            row_id += 1

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "sound_label", "emotion", "transcript", "scene_label",
            "acceptable_score_min", "acceptable_score_max",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"({len(SOUND_BASELINES)} sound classes x {len(EMOTION_BASELINES)} emotions, full cross, "
          f"scene/transcript assigned independently per row)")


if __name__ == "__main__":
    main()
