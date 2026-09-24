"""Unit/integration tests for api.py.

api.py loads four heavy models at import time, so they are replaced with
stubs in sys.modules before `import api` -- the tests never touch torch or
real model weights.
"""
import io
import struct
import sys
import types
import wave

import pytest


def _make_stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture
def api_module(monkeypatch):
    """Imports api.py with all 4 model-loading submodules stubbed out,
    then yields the module. Each call gets a *fresh* import so tests
    don't leak stub state into one another."""
    # NOTE: api.py calls these as classify_x(bytes, model) — bytes first,
    # model second — so the stubs must match that argument order exactly.
    monkeypatch.setitem(
        sys.modules, "models.efficientat_classifier",
        _make_stub_module(
            "models.efficientat_classifier",
            load_efficientat_model=lambda: "STUB_EFFICIENTAT_MODEL",
            classify_audio=lambda wav_bytes, model: {"label": "traffic noise", "confidence": 0.91},
        ),
    )
    monkeypatch.setitem(
        sys.modules, "models.transcriber",
        _make_stub_module(
            "models.transcriber",
            load_transcription_model=lambda: "STUB_TRANSCRIBER_MODEL",
            transcribe_audio=lambda wav_bytes, model: "this is a stubbed transcript",
        ),
    )
    monkeypatch.setitem(
        sys.modules, "models.emotion_classifier",
        _make_stub_module(
            "models.emotion_classifier",
            load_emotion_model=lambda: "STUB_EMOTION_MODEL",
            classify_emotion_from_audio=lambda wav_bytes, model: {"label": "Calm", "confidence": 0.82},
        ),
    )
    monkeypatch.setitem(
        sys.modules, "models.scene_classifier",
        _make_stub_module(
            "models.scene_classifier",
            load_scene_model=lambda: "STUB_SCENE_MODEL",
            classify_scene_from_image=lambda image_bytes, model: {"scene": "park", "confidence": 0.73},
        ),
    )

    # api.py also touches models.db at import time (db.init_db()) — point
    # it at a throwaway file so these tests never write to urbanpulse.db.
    import os
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(tmp_path)
    monkeypatch.setenv("_UNIT_TEST_DB_PATH", tmp_path)

    sys.modules.pop("api", None)
    sys.modules.pop("models.db", None)
    import models.db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path)

    import api as api_mod
    monkeypatch.setattr(api_mod.db, "DB_PATH", tmp_path)
    api_mod.db.init_db()

    yield api_mod

    sys.modules.pop("api", None)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture
def client(api_module):
    api_module.app.testing = True
    return api_module.app.test_client()


def _silent_wav_bytes(duration_s=0.1, framerate=16000):
    """A minimal valid mono 16-bit PCM WAV file of silence, built with the
    stdlib `wave` module — no ffmpeg/pydub dependency for the test input
    itself."""
    n_frames = int(duration_s * framerate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(struct.pack("<%dh" % n_frames, *([0] * n_frames)))
    return buf.getvalue()


# ---------------------------------------------------------------------
# convert_to_wav_bytes
# ---------------------------------------------------------------------

def test_convert_to_wav_bytes_resamples_to_requested_rate(api_module):
    original = _silent_wav_bytes(framerate=44100)
    converted = api_module.convert_to_wav_bytes(original, sample_rate=16000)

    with wave.open(io.BytesIO(converted), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1


def test_convert_to_wav_bytes_falls_back_to_raw_bytes_on_bad_input(api_module):
    garbage = b"this is not audio data at all"
    result = api_module.convert_to_wav_bytes(garbage, sample_rate=16000)
    # Documented fallback behaviour: on a conversion failure, the original
    # bytes are returned unchanged rather than raising.
    assert result == garbage


# ---------------------------------------------------------------------
# /api/analyze and /api/scene (model calls fully stubbed)
# ---------------------------------------------------------------------

def test_analyze_route_returns_stubbed_classification(client):
    wav_bytes = _silent_wav_bytes()
    res = client.post(
        "/api/analyze",
        data={"audio": (io.BytesIO(wav_bytes), "clip.wav")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "success"
    assert body["results"] == {"label": "traffic noise", "confidence": 0.91}


def test_analyze_route_rejects_request_with_no_audio(client):
    res = client.post("/api/analyze", json={})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_analyze_route_returns_500_when_model_failed_to_load(api_module):
    api_module.efficientat_model = None
    client = api_module.app.test_client()
    wav_bytes = _silent_wav_bytes()
    res = client.post(
        "/api/analyze",
        data={"audio": (io.BytesIO(wav_bytes), "clip.wav")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 500


def test_scene_route_returns_stubbed_scene_and_confidence(client):
    fake_image_bytes = b"\xff\xd8\xff\xe0not a real jpeg but that's fine, it's stubbed"
    res = client.post(
        "/api/scene",
        data={"image": (io.BytesIO(fake_image_bytes), "photo.jpg")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "success"
    assert body["scene"] == "park"
    assert body["confidence"] == 0.73


# ---------------------------------------------------------------------
# /api/sound_hunt/* routes (exercise the Flask app + db together)
# ---------------------------------------------------------------------

def test_sound_hunt_completions_starts_empty(client):
    res = client.get("/api/sound_hunt/completions")
    assert res.status_code == 200
    assert res.get_json()["completed"] == []


def test_completing_a_challenge_is_reflected_in_completions(client):
    # /api/sound_hunt/complete now re-classifies the submitted audio
    # server-side and only completes the challenge if it genuinely
    # matches (see models/sound_hunt.py) -- the stubbed classifier always
    # returns "traffic noise", so "traffic" is the challenge that matches.
    wav_bytes = _silent_wav_bytes()
    res = client.post(
        "/api/sound_hunt/complete",
        data={"challenge_id": "traffic", "audio": (io.BytesIO(wav_bytes), "clip.wav")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    assert res.get_json()["verified"] is True
    # mark_challenge_complete() returns the full updated completions list.
    assert res.get_json()["completed"] == ["traffic"]

    res = client.get("/api/sound_hunt/completions")
    assert res.get_json()["completed"] == ["traffic"]


def test_completing_a_non_matching_challenge_is_rejected(client):
    # "rain" doesn't match the stub's "traffic noise" label -- the server
    # must reject this rather than blindly trusting the challenge_id.
    wav_bytes = _silent_wav_bytes()
    res = client.post(
        "/api/sound_hunt/complete",
        data={"challenge_id": "rain", "audio": (io.BytesIO(wav_bytes), "clip.wav")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert res.get_json()["verified"] is False

    res = client.get("/api/sound_hunt/completions")
    assert "rain" not in res.get_json()["completed"]


def test_completing_without_audio_is_rejected(client):
    # The exploit this fix closes: a bare challenge_id with no audio must
    # no longer be accepted unconditionally.
    res = client.post("/api/sound_hunt/complete", json={"challenge_id": "traffic"})
    assert res.status_code == 400


def test_completing_the_same_challenge_twice_does_not_duplicate(client):
    wav_bytes = _silent_wav_bytes()
    for _ in range(2):
        client.post(
            "/api/sound_hunt/complete",
            data={"challenge_id": "traffic", "audio": (io.BytesIO(wav_bytes), "clip.wav")},
            content_type="multipart/form-data",
        )

    res = client.get("/api/sound_hunt/completions")
    assert res.get_json()["completed"].count("traffic") == 1
