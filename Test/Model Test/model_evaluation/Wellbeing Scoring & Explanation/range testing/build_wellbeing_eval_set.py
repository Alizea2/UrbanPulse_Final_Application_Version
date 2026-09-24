"""Builds the curated ground-truth evaluation set for the original 2-input
Step 4 scorer (sound + emotion). Superseded by the 4-input builder.
"""

import csv
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Wellbeing Scoring & Explanation/range testing
TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
OUT_PATH = os.path.join(TESTS_ROOT, "Datasets", "Wellbeing Scoring & Explanation", "wellbeing_eval_set.csv")

# How far a predicted score may land from the center score and still
# count as correct. +/-0.10 was chosen as a reasonably tight band given
# scores span the full [0.0, 1.0] range in only ~0.05 increments here.
TOLERANCE = 0.10

# Baselines per SONYC-UST's 8 coarse categories (Cartwright et al., 2019).
# The category value is what the cited research grounds; the within-category
# differences are small literature-justified nudges.
SOUND_CATEGORY_BASELINES = {
    # 1. ENGINE — WHO (2018) dose-response: annoyance rises with source size.
    "small-sounding-engine": 0.50,
    "medium-sounding-engine": 0.40,
    "large-sounding-engine": 0.28,

    # 2. MACHINERY IMPACT — impulsive-noise annoyance penalty (WHO, 2018;
    # Miedema & Vos, 1998).
    "rock-drill": 0.10,
    "jackhammer": 0.08,
    "hoe-ram": 0.10,
    "pile-driver": 0.08,

    # 3. NON-MACHINERY IMPACT — same penalty as category 2, less intense.
    "non-machinery-impact": 0.35,

    # 4. POWERED SAW — impulsive plus a tonal penalty (ISO 1996-2).
    "chainsaw": 0.15,
    "small-medium-rotating-saw": 0.20,
    "large-rotating-saw": 0.15,

    # 5. ALERT SIGNAL — designed to be maximally aversive (Kryter; ISO 12913).
    "car-horn": 0.20,
    "car-alarm": 0.15,
    "siren": 0.10,
    "reverse-beeper": 0.25,

    # 6. MUSIC — rated more pleasant than mechanical sources at comparable
    # levels (Axelsson et al., 2010; Yang & Kang, 2005). The ice-cream-truck
    # nudge is qualitative (Schafer's "soundmark"), not a measured finding.
    "stationary-music": 0.60,
    "mobile-music": 0.55,
    "ice-cream-truck": 0.65,

    # 7. HUMAN VOICE — positive at moderate levels, falling as loudness and
    # crowd size rise (Axelsson et al., 2010). Amplified speech sits apart.
    "person-or-small-group-talking": 0.65,
    "person-or-small-group-shouting": 0.25,
    "large-crowd": 0.40,
    "amplified-speech": 0.35,

    # 8. DOG — a recognised but moderate residential annoyance category.
    "dog-barking-whining": 0.35,
}

# Flat (sound_label, baseline) list over the 23 SONYC-UST fine classes.
SOUND_BASELINES = list(SOUND_CATEGORY_BASELINES.items())

# (emotion, baseline). Ordering follows Russell's circumplex; the exact
# magnitudes are a calibrated judgment call, not independently validated.
EMOTION_BASELINES = [
    ("Calm", 0.85),
    ("Content", 0.90),
    ("Anxious", 0.30),
    ("Stressed", 0.15),
    ("Annoyed", 0.25),
]

# Short plausible transcripts, cycled across the rows of each emotion.
TRANSCRIPT_POOL = {
    "Calm": [
        "I'm just sitting outside enjoying some fresh air.",
        "This spot is really peaceful right now.",
        "Taking a slow walk, nothing much going on.",
        "It's a quiet afternoon, feels nice out here.",
    ],
    "Content": [
        "Had a great coffee and I'm feeling good today.",
        "This is honestly a really nice moment right now.",
        "Everything's going well, I'm in a good mood.",
        "Just relaxing and enjoying the day so far.",
    ],
    "Anxious": [
        "I don't know, I'm feeling a bit on edge right now.",
        "Something about this street is making me uneasy.",
        "I keep looking around, not sure why I'm nervous.",
        "My heart's racing a little, not sure what's going on.",
    ],
    "Stressed": [
        "I'm so behind on everything today, this is a lot.",
        "I really need this to be over, I can't focus.",
        "Everything is piling up and I just need a minute.",
        "I'm running late and nothing is going right.",
    ],
    "Annoyed": [
        "Honestly this is getting really irritating now.",
        "Can this just stop already, I've had enough.",
        "I'm getting pretty fed up with this whole thing.",
        "This is so annoying, I just want some quiet.",
    ],
}


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    rows = []
    row_id = 1
    for sound_label, sound_score in SOUND_BASELINES:
        for emotion, emotion_score in EMOTION_BASELINES:
            transcript_options = TRANSCRIPT_POOL[emotion]
            transcript = transcript_options[row_id % len(transcript_options)]

            center_score = round(0.5 * sound_score + 0.5 * emotion_score, 2)
            range_min = max(0.0, round(center_score - TOLERANCE, 2))
            range_max = min(1.0, round(center_score + TOLERANCE, 2))

            rows.append({
                "id": row_id,
                "sound_label": sound_label,
                "emotion": emotion,
                "transcript": transcript,
                "acceptable_score_min": f"{range_min:.2f}",
                "acceptable_score_max": f"{range_max:.2f}",
            })
            row_id += 1

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "sound_label", "emotion", "transcript",
            "acceptable_score_min", "acceptable_score_max",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"({len(SOUND_BASELINES)} sound classes x {len(EMOTION_BASELINES)} emotions, full cross)")


if __name__ == "__main__":
    main()
