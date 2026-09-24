import sqlite3
import os
import time
import math

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "urbanpulse.db")

# A new pin within this radius replaces the existing one, so the map shows
# the current read on a location rather than every historical report.
REPLACE_RADIUS_M = 80


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the shared tables if they don't exist yet."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diary_entries (
            id TEXT PRIMARY KEY,
            timestamp INTEGER NOT NULL,
            sound_label TEXT,
            scene_label TEXT,
            transcript TEXT,
            emotion_label TEXT,
            emotion_confidence REAL,
            wellbeing_score REAL,
            wellbeing_explanation TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS map_pins (
            id TEXT PRIMARY KEY,
            timestamp INTEGER NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            emotion TEXT,
            desc TEXT,
            scene_label TEXT,
            color TEXT,
            wellbeing_score REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sound_hunt_completions (
            challenge_id TEXT PRIMARY KEY,
            completed_at INTEGER NOT NULL
        )
    """)
    # CREATE TABLE IF NOT EXISTS skips existing tables, so an older DB file
    # needs the scene_label column added directly.
    for table in ("diary_entries", "map_pins"):
        existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "scene_label" not in existing_columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN scene_label TEXT")
    conn.commit()
    conn.close()


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def save_diary_entry(entry):
    conn = get_connection()
    entry_id = entry.get("id") or str(int(time.time() * 1000))
    timestamp = entry.get("timestamp") or int(time.time() * 1000)
    emotion = entry.get("emotion") or {}
    conn.execute(
        """INSERT INTO diary_entries
           (id, timestamp, sound_label, scene_label, transcript, emotion_label, emotion_confidence,
            wellbeing_score, wellbeing_explanation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (entry_id, timestamp, entry.get("soundLabel"), entry.get("sceneLabel"), entry.get("transcript"),
         emotion.get("label"), emotion.get("confidence"),
         entry.get("wellbeingScore"), entry.get("wellbeingExplanation")),
    )
    conn.commit()
    conn.close()
    return get_diary_entry(entry_id)


def get_diary_entry(entry_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM diary_entries WHERE id = ?", (entry_id,)).fetchone()
    conn.close()
    return _row_to_diary_entry(row) if row else None


def get_diary_entries():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM diary_entries ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [_row_to_diary_entry(r) for r in rows]


def _row_to_diary_entry(row):
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "soundLabel": row["sound_label"],
        "sceneLabel": row["scene_label"],
        "transcript": row["transcript"],
        "emotion": {"label": row["emotion_label"], "confidence": row["emotion_confidence"]} if row["emotion_label"] else None,
        "wellbeingScore": row["wellbeing_score"],
        "wellbeingExplanation": row["wellbeing_explanation"],
    }


def save_map_pin(pin):
    """Inserts a pin, first deleting any existing pin within REPLACE_RADIUS_M."""
    conn = get_connection()
    pin_id = pin.get("id") or str(int(time.time() * 1000))
    timestamp = pin.get("timestamp") or int(time.time() * 1000)
    loc = pin.get("loc") or {}
    lat, lon = loc.get("latitude"), loc.get("longitude")

    existing = conn.execute("SELECT id, latitude, longitude FROM map_pins").fetchall()
    for row in existing:
        if _haversine_m(lat, lon, row["latitude"], row["longitude"]) <= REPLACE_RADIUS_M:
            conn.execute("DELETE FROM map_pins WHERE id = ?", (row["id"],))

    conn.execute(
        """INSERT INTO map_pins (id, timestamp, latitude, longitude, emotion, desc, scene_label, color, wellbeing_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pin_id, timestamp, lat, lon, pin.get("emotion"), pin.get("desc"), pin.get("sceneLabel"),
         pin.get("color"), pin.get("wellbeingScore")),
    )
    conn.commit()
    conn.close()
    return get_map_pin(pin_id)


def get_map_pin(pin_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM map_pins WHERE id = ?", (pin_id,)).fetchone()
    conn.close()
    return _row_to_map_pin(row) if row else None


def get_map_pins():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM map_pins ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [_row_to_map_pin(r) for r in rows]


def _row_to_map_pin(row):
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "loc": {"latitude": row["latitude"], "longitude": row["longitude"]},
        "emotion": row["emotion"],
        "desc": row["desc"],
        "sceneLabel": row["scene_label"],
        "color": row["color"],
        "wellbeingScore": row["wellbeing_score"],
    }


def get_completed_challenges():
    conn = get_connection()
    rows = conn.execute("SELECT challenge_id FROM sound_hunt_completions").fetchall()
    conn.close()
    return [r["challenge_id"] for r in rows]


def mark_challenge_complete(challenge_id):
    """Idempotent: re-completing a challenge keeps its original timestamp."""
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO sound_hunt_completions (challenge_id, completed_at) VALUES (?, ?)",
        (challenge_id, int(time.time() * 1000)),
    )
    conn.commit()
    conn.close()
    return get_completed_challenges()
