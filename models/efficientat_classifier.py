"""Sound classification via EfficientAT (mn10_as), loaded in-process.

Source: github.com/fschmid56/EfficientAT (MIT). It has no pip package, so
the required files are vendored under models/efficientat_src/.
"""
import contextlib
import io
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# The vendored source has to be importable before the model loads.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "efficientat_src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

SAMPLE_RATE = 32000          # the rate EfficientAT expects
TOP_N = 3                    # labels returned to the app
LOW_CONFIDENCE_THRESHOLD = 0.15   # below this, the result is flagged unsure


def clean_label(raw_label):
    """Takes the first term from AudioSet's comma-separated synonym lists."""
    return raw_label.split(",")[0].strip()


def load_efficientat_model():
    """Loads the model and its mel preprocessor once, kept resident."""
    from efficientat_mn_pkg.mn.model import get_model as get_mobilenet
    from efficientat_mn_pkg.preprocess import AugmentMelSTFT
    from efficientat_helpers.utils import labels

    with contextlib.redirect_stdout(io.StringIO()):  # suppress the model's architecture dump
        model = get_mobilenet(width_mult=1.0, pretrained_name="mn10_as")
    model.eval()

    mel = AugmentMelSTFT(n_mels=128, sr=SAMPLE_RATE, win_length=800, hopsize=320)
    mel.eval()

    return model, mel, labels


def classify_audio(audio_bytes, model_bundle):
    """Runs EfficientAT on a WAV clip and returns the top-3 labels,
    in the same shape the app expects (label / raw_label / confidence)."""
    import tempfile
    import torch
    import librosa
    import numpy as np

    model, mel, class_names = model_bundle

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
        # Clips under a second are padded, as the model needs a full window.
        min_samples = int(sr * 1.0)
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)))

        with torch.no_grad():
            waveform = torch.from_numpy(y[None, :]).float()
            spec = mel(waveform)   # waveform -> mel spectrogram
            preds, _ = model(spec.unsqueeze(0))
            # Sigmoid, not softmax: AudioSet is multi-label, so several
            # sounds can be present at once and the scores need not sum to 1.
            scores = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:TOP_N]   # highest first
        results = [
            {
                "label": clean_label(class_names[i]),
                "raw_label": class_names[i],
                "confidence": float(scores[i]),
            }
            for i in top_indices
        ]

        if results and results[0]["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            results[0]["low_confidence"] = True

        return results
    finally:
        os.remove(temp_path)   # the temp file goes even if inference raised
