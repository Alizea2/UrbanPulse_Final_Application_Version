"""Tests how diary_entries and map_pins query latency changes as the tables
grow from empty to 10k+ rows.
"""
import os
import sys
import time
import random
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import db

SIZES = [10, 100, 1000, 10000]
SOUND_LABELS = ["Traffic noise", "Birdsong", "Crowd noise", "Construction", "Silence"]
EMOTIONS = ["Calm", "Content", "Anxious", "Stressed", "Annoyed"]


def seed_diary_entries(n, offset=0):
    conn = db.get_connection()
    rows = []
    for j in range(n):
        i = offset + j
        rows.append((
            f"seed-{i}",
            1700000000000 + i,
            random.choice(SOUND_LABELS),
            "park",
            f"test transcript {i}",
            random.choice(EMOTIONS),
            0.8,
            round(random.random(), 2),
            "test explanation",
        ))
    conn.executemany(
        """INSERT INTO diary_entries
           (id, timestamp, sound_label, scene_label, transcript, emotion_label, emotion_confidence,
            wellbeing_score, wellbeing_explanation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()


def seed_map_pins(n, offset=0):
    conn = db.get_connection()
    rows = []
    for j in range(n):
        i = offset + j
        # Spread pins out globally so save_map_pin's proximity-replace logic
        # doesn't collapse them during seeding.
        lat = -60 + (i % 1000) * 0.1
        lon = -170 + (i // 1000) * 0.1
        rows.append((
            f"pin-{i}", 1700000000000 + i, lat, lon,
            random.choice(EMOTIONS), "test pin", "park", "#6E8B57", round(random.random(), 2),
        ))
    conn.executemany(
        """INSERT INTO map_pins (id, timestamp, latitude, longitude, emotion, desc, scene_label, color, wellbeing_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()


def time_it(fn, n=5):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def main():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    db.DB_PATH = path
    db.init_db()

    print(f"{'Rows':>8} {'get_diary_entries (s)':>25} {'get_map_pins (s)':>20}")
    prev_diary = 0
    prev_pins = 0
    for size in SIZES:
        seed_diary_entries(size - prev_diary, offset=prev_diary)
        seed_map_pins(size - prev_pins, offset=prev_pins)
        prev_diary = size
        prev_pins = size

        diary_time = time_it(db.get_diary_entries)
        pins_time = time_it(db.get_map_pins)
        print(f"{size:>8} {diary_time:>25.4f} {pins_time:>20.4f}")

    os.remove(path)


if __name__ == "__main__":
    main()
