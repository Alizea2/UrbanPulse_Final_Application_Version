"""Text-based emotion classifier (GoEmotions) used only by test_goemotion_ravdess.py.

Not part of the app: GoEmotions lost the Step 3 comparison and was never adopted.
Model: SamLowe/roberta-base-go_emotions.
"""

from transformers import pipeline

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# GoEmotions' 28 labels -> UrbanPulse's five wellbeing states.
EMOTION_MAP = {
    # --- Calm ---
    "neutral": "Calm",
    "relief": "Calm",
    "realization": "Calm",
    # --- Content ---
    "admiration": "Content",
    "amusement": "Content",
    "approval": "Content",
    "caring": "Content",
    "curiosity": "Content",
    "desire": "Content",
    "excitement": "Content",
    "gratitude": "Content",
    "joy": "Content",
    "love": "Content",
    "optimism": "Content",
    "pride": "Content",
    # --- Anxious ---
    "confusion": "Anxious",
    "disappointment": "Anxious",
    "embarrassment": "Anxious",
    "fear": "Anxious",
    "grief": "Anxious",
    "nervousness": "Anxious",
    "remorse": "Anxious",
    "sadness": "Anxious",
    "surprise": "Anxious",
    # --- Stressed ---
    "anger": "Stressed",
    # --- Annoyed ---
    "annoyance": "Annoyed",
    "disapproval": "Annoyed",
    "disgust": "Annoyed",
}

DEFAULT_STATE = "Calm"

_classifier = None


def load_goemotion_model():
    """Loads and caches the pipeline."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,
        )
    return _classifier


def classify_emotion(transcript):
    """Returns the top GoEmotions label, its UrbanPulse state, and confidence."""
    classifier = load_goemotion_model()

    text = (transcript or "").strip()
    if not text:
        return {"emotion": DEFAULT_STATE, "raw_label": "neutral", "confidence": 0.0}

    scores = classifier(text)[0]
    top = max(scores, key=lambda s: s["score"])
    raw_label = top["label"]

    return {
        "emotion": EMOTION_MAP.get(raw_label, DEFAULT_STATE),
        "raw_label": raw_label,
        "confidence": float(top["score"]),
    }


if __name__ == "__main__":
    load_goemotion_model()
    for sentence in ("Kids are talking by the door.", "Dogs are sitting by the door."):
        print(sentence, "->", classify_emotion(sentence))
