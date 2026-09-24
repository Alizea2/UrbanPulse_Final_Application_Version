"""Server-side Sound Hunt challenge definitions and verification.

Mirrors the CHALLENGES list in mobile_app/src/screens/SoundHuntScreen.js --
keep the two in sync, as both enforce the same rules independently.
"""
import re
from datetime import datetime

CHALLENGES = {
    # Sound only -- match the keyword at any time of day.
    "rain": {"keywords": ["rain"], "time": None},
    "silence": {"keywords": ["silence"], "time": None},
    "cafe": {"keywords": ["speech", "chatter", "dish", "cup", "clink"], "time": None},
    "traffic": {"keywords": ["traffic", "vehicle", "engine", "car"], "time": None},
    "music": {"keywords": ["music"], "time": None},
    "dog": {"keywords": ["dog", "bark"], "time": None},

    # Time only -- any recording counts inside the window.
    "early_riser": {"keywords": None, "time": {"start": 5, "end": 8}},
    "night_owl": {"keywords": None, "time": {"start": 22, "end": 2}},

    # Both -- the sound must match AND fall inside the window.
    "dawn_chorus": {"keywords": ["bird"], "time": {"start": 5, "end": 11}},
    "midnight_quiet": {"keywords": ["silence"], "time": {"start": 22, "end": 4}},
    "coffee_rush": {"keywords": ["speech", "chatter", "dish", "cup", "clink"], "time": {"start": 7, "end": 11}},
    "rush_hour": {"keywords": ["traffic", "vehicle", "engine", "car"], "time": {"start": 17, "end": 19}},
}


def matches_sound(results, keywords):
    """True if any classify_audio() label contains a keyword as a whole word."""
    if keywords is None:
        return True  # time-only challenge -- any recording counts
    for r in results or []:
        label = (r.get("label") or "").lower()
        if any(re.search(rf"\b{re.escape(k)}\b", label) for k in keywords):
            return True
    return False


def is_within_time_window(time_window, now=None):
    if time_window is None:
        return True
    # Uses the server's clock, not a timestamp the client sent.
    now = now or datetime.now()
    hour = now.hour
    start, end = time_window["start"], time_window["end"]
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end  # wraps past midnight


def verify_challenge(challenge_id, classification_results):
    """Returns (ok, reason). Re-checks sound and time server-side; the
    client's own claim is not trusted."""
    # An unknown id fails closed rather than defaulting to success.
    challenge = CHALLENGES.get(challenge_id)
    if challenge is None:
        return False, f"Unknown challenge_id: {challenge_id!r}"
    if not matches_sound(classification_results, challenge["keywords"]):
        return False, "Recording does not match the challenge's required sound"
    if not is_within_time_window(challenge["time"]):
        return False, "Recording was not made within the challenge's time window"
    return True, "ok"
