import tempfile
import os

MODEL_NAME = "Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition"

# The checkpoint stores its classifier head under older parameter names;
# without this rename the head loads randomly initialised, in silence.
_KEY_RENAME = {
    "classifier.dense.weight": "projector.weight",
    "classifier.dense.bias": "projector.bias",
    "classifier.out_proj.weight": "classifier.weight",
    "classifier.out_proj.bias": "classifier.bias",
}

# CREMA-D's 6 native label codes -> UrbanPulse's 5 wellbeing states.
LABEL_TO_URBANPULSE = {
    "ANG": "Stressed", "DIS": "Annoyed", "FEA": "Anxious",
    "SAD": "Anxious", "HAP": "Content", "NEU": "Calm",
}

# Anxious is the only state fed by two labels (FEA and SAD), so it is
# structurally over-predicted. This offset corrects for that.
FEA_SAD_LOGIT_BIAS = -2.0


def load_emotion_model():
    """Loads the Yassmen wav2vec2 speech-emotion model and its extractor."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForSequenceClassification, AutoConfig, AutoFeatureExtractor

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.classifier_proj_size = 1024
    model = Wav2Vec2ForSequenceClassification(config)

    ckpt_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    state_dict = load_file(ckpt_path)
    fixed_state_dict = {_KEY_RENAME.get(k, k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Yassmen checkpoint did not load cleanly — missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, extractor


def classify_emotion_from_audio(audio_bytes, model):
    """Maps the model's top prediction onto one of the 5 wellbeing states.

    `model` is the (model, extractor) tuple from load_emotion_model()."""
    import torch
    import librosa

    ser_model, extractor = model

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        y, sr = librosa.load(temp_path, sr=16000, mono=True)
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = ser_model(inputs.input_values).logits

        # Applied before argmax: this is the decision rule, not a display tweak.
        label2id = ser_model.config.label2id
        logits[0, label2id["FEA"]] += FEA_SAD_LOGIT_BIAS
        logits[0, label2id["SAD"]] += FEA_SAD_LOGIT_BIAS

        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs))
        raw_label = ser_model.config.id2label[top_idx]

        return {
            "emotion": LABEL_TO_URBANPULSE.get(raw_label, "Calm"),
            "raw_label": raw_label,
            "confidence": float(probs[top_idx]),
        }
    finally:
        os.remove(temp_path)
