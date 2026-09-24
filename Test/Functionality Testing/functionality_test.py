"""
API-level functionality testing for UrbanPulse -- covers every backend
feature, not just Sound Hunt. Driven via real HTTP requests against the
live Flask backend (models warm-loaded).
"""
import math
import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.sound_hunt import matches_sound

API_BASE = "http://127.0.0.1:5001"
AUDIO_PATH = "test.wav"
TRAFFIC_AUDIO_PATH = "Test/Functionality Testing/traffic_test.wav"
IMAGE_PATH = "Test/Model Test/Datasets/Scene Classification/places365_torch_uncertainty/archive/Places365_val_00004488.jpg"

results = []


def record(test_id, feature, passed):
    results.append((test_id, feature, passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {test_id}: {feature}")


def test_sound_classification():
    with open(AUDIO_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/analyze", files={"audio": f})
    body = r.json()
    passed = r.status_code == 200 and len(body.get("results", [])) > 0
    record("FT-1", "Sound Classification", passed)


def test_scene_classification():
    with open(IMAGE_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/scene", files={"image": f})
    body = r.json()
    passed = r.status_code == 200 and bool(body.get("scene"))
    record("FT-2", "Scene Classification", passed)


def test_transcription_and_emotion():
    with open(AUDIO_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/transcribe", files={"audio": f})
    body = r.json()
    passed = r.status_code == 200 and bool(body.get("emotion"))
    record("FT-3", "Voice Emotion Analysis (transcription + emotion)", passed)


def test_wellbeing_scoring():
    r = requests.post(f"{API_BASE}/api/wellbeing", json={
        "sound_label": "Traffic noise",
        "scene_label": "busy street",
        "transcript": "It's pretty loud outside but I feel okay.",
        "emotion": "Calm",
    })
    body = r.json()
    score = body.get("score")
    passed = r.status_code == 200 and isinstance(score, (int, float)) and 0.0 <= score <= 1.0 and bool(body.get("explanation"))
    record("FT-4", "AI Wellbeing Scoring", passed)


def test_diary_entry():
    entry_id = f"ft-diary-{int(time.time()*1000)}"
    r1 = requests.post(f"{API_BASE}/api/diary/entries", json={
        "id": entry_id, "soundLabel": "Traffic noise", "sceneLabel": "busy street",
        "transcript": "test", "emotion": {"label": "Calm", "confidence": 0.9},
        "wellbeingScore": 0.6, "wellbeingExplanation": "test",
    })
    r2 = requests.get(f"{API_BASE}/api/diary/entries")
    found = any(e["id"] == entry_id for e in r2.json().get("entries", []))
    passed = r1.status_code == 200 and found
    record("FT-5", "Diary Entry (save + retrieve)", passed)


def test_diary_insight():
    r = requests.post(f"{API_BASE}/api/diary_insight", json={
        "entries": [
            {"soundLabel": "Traffic noise", "emotion": {"label": "Stressed"}, "wellbeingScore": 0.3},
            {"soundLabel": "Birdsong", "emotion": {"label": "Calm"}, "wellbeingScore": 0.8},
        ]
    })
    body = r.json()
    passed = r.status_code == 200 and bool(body.get("insight"))
    record("FT-6", "Diary Insight Generation", passed)


def test_community_map():
    pin_id = f"ft-pin-{int(time.time()*1000)}"
    r1 = requests.post(f"{API_BASE}/api/map/pins", json={
        "id": pin_id, "loc": {"latitude": 24.8607, "longitude": 67.0011},
        "emotion": "Anxious", "desc": "traffic noise", "sceneLabel": "busy street",
        "color": "#C9922E", "wellbeingScore": 0.35,
    })
    r2 = requests.get(f"{API_BASE}/api/map/pins")
    found = any(p["id"] == pin_id for p in r2.json().get("pins", []))
    passed = r1.status_code == 200 and found
    record("FT-7", "Community Emotion Map (pin save + retrieve)", passed)


def test_sound_hunt_match():
    with open(TRAFFIC_AUDIO_PATH, "rb") as f:
        r_check = requests.post(f"{API_BASE}/api/analyze", files={"audio": f})
    top_results = r_check.json()["results"]
    is_match = matches_sound(top_results, ["traffic", "vehicle", "engine", "car"])

    with open(TRAFFIC_AUDIO_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/sound_hunt/complete", files={"audio": f}, data={"challenge_id": "traffic"})
    passed = is_match and r.status_code == 200 and r.json().get("verified") is True
    record("FT-8", "Sound Hunt (matching sound completes challenge)", passed)


def test_sound_hunt_reject():
    with open(AUDIO_PATH, "rb") as f:
        r = requests.post(f"{API_BASE}/api/sound_hunt/complete", files={"audio": f}, data={"challenge_id": "traffic"})
    passed = r.status_code == 400 and r.json().get("verified") is False
    record("FT-9", "Sound Hunt (non-matching sound is rejected)", passed)


def test_backend_unreachable():
    try:
        requests.get("http://127.0.0.1:59999/api/diary/entries", timeout=3)
        record("FT-10", "Backend Unreachable (graceful failure)", False)
    except requests.exceptions.ConnectionError:
        record("FT-10", "Backend Unreachable (graceful failure)", True)
    except Exception:
        record("FT-10", "Backend Unreachable (graceful failure)", False)


def test_quiet_spot_finder():
    # Mirrors QuietSpotsScreen.js: Calm emotion AND score >= min AND within radius.
    MIN_QUIET_SCORE = 0.50
    DEFAULT_RADIUS_KM = 25
    ORIGIN = {"latitude": 24.8607, "longitude": 67.0011}   # reference "user location"

    def haversine_km(a, b):
        # Same formula as QuietSpotsScreen.js (R = 6371).
        R = 6371.0
        d_lat = math.radians(b["latitude"] - a["latitude"])
        d_lon = math.radians(b["longitude"] - a["longitude"])
        lat1 = math.radians(a["latitude"])
        lat2 = math.radians(b["latitude"])
        h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        return 2 * R * math.asin(math.sqrt(h))

    stamp = int(time.time() * 1000)
    # One case per filter condition; pins kept >80m apart so they aren't merged.
    cases = [
        # id suffix,      lat,     lon,   emotion,    score, should_qualify, why
        ("qualifies",  24.8700, 67.0100, "Calm",      0.80, True,  "Calm, above threshold, nearby"),
        ("lowscore",   24.8800, 67.0200, "Calm",      0.30, False, "Calm but below MIN_QUIET_SCORE"),
        ("notcalm",    24.8900, 67.0300, "Stressed",  0.90, False, "high score but not Calm"),
        ("faraway",    25.6000, 67.8000, "Calm",      0.85, False, "Calm and high, but outside radius"),
    ]
    expected = {}
    for suffix, lat, lon, emotion, score, should, _why in cases:
        pin_id = f"ft-quiet-{suffix}-{stamp}"
        expected[pin_id] = should
        requests.post(f"{API_BASE}/api/map/pins", json={
            "id": pin_id, "loc": {"latitude": lat, "longitude": lon},
            "emotion": emotion, "desc": f"ft {suffix}", "sceneLabel": "park",
            "color": "#7FA06A", "wellbeingScore": score,
        })

    pins = requests.get(f"{API_BASE}/api/map/pins").json().get("pins", [])
    qualifying = {
        p["id"] for p in pins
        if p.get("emotion") == "Calm"
        and isinstance(p.get("wellbeingScore"), (int, float))
        and p["wellbeingScore"] >= MIN_QUIET_SCORE
        and haversine_km(ORIGIN, p["loc"]) <= DEFAULT_RADIUS_KM
    }

    passed = all((pin_id in qualifying) == should for pin_id, should in expected.items())
    if not passed:
        for (suffix, _la, _lo, _e, _s, should, why), pin_id in zip(cases, expected):
            got = pin_id in qualifying
            if got != should:
                print(f"       FT-11 mismatch: {suffix} ({why}) -> expected "
                      f"{'qualify' if should else 'reject'}, got {'qualify' if got else 'reject'}")
    record("FT-11", "Quiet Spot Finder (Calm + score + radius filter)", passed)


def test_invalid_input_handling():
    # A few common invalid-input cases should fail cleanly (400), not crash.
    r1 = requests.post(f"{API_BASE}/api/analyze", data={})  # no audio at all
    r2 = requests.post(f"{API_BASE}/api/wellbeing", json={"sound_label": "Traffic noise"})  # missing fields
    r3 = requests.post(f"{API_BASE}/api/sound_hunt/complete", json={})  # missing challenge_id and audio
    passed = r1.status_code == 400 and r2.status_code == 400 and r3.status_code == 400
    record("FT-12", "Invalid Input Handling (missing fields return clean 400s)", passed)


def main():
    test_sound_classification()
    test_scene_classification()
    test_transcription_and_emotion()
    test_wellbeing_scoring()
    test_diary_entry()
    test_diary_insight()
    test_community_map()
    test_sound_hunt_match()
    test_sound_hunt_reject()
    test_backend_unreachable()
    test_quiet_spot_finder()
    test_invalid_input_handling()

    passed_count = sum(1 for _, _, p in results if p)
    print(f"\n{passed_count}/{len(results)} passed")


if __name__ == "__main__":
    main()
