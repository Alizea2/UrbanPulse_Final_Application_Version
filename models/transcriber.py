import tempfile
import os

MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"

def load_transcription_model():
    """Loads wav2vec2-large-960h-lv60-self for local transcription."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

def transcribe_audio(audio_bytes, model):
    """Returns the transcript of the audio bytes.

    `model` is the (processor, model) tuple from load_transcription_model()."""
    import torch
    import librosa

    processor, wav2vec2_model = model

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        y, sr = librosa.load(temp_path, sr=16000, mono=True)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = wav2vec2_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        raw_text = processor.batch_decode(predicted_ids)[0]

        # wav2vec2 outputs unpunctuated ALL CAPS; make it read as a sentence.
        text = raw_text.strip()
        if text:
            text = text[0].upper() + text[1:].lower()
        return text
    finally:
        os.remove(temp_path)
