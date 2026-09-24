"""Unit tests for models/db.py: _haversine_m() and save_map_pin()'s
replace-within-REPLACE_RADIUS_M rule.

Each test runs against a fresh temporary SQLite file, never urbanpulse.db.
"""
import math
import os
import tempfile

import pytest

from models import db


@pytest.fixture
def temp_db(monkeypatch):
    """Points models.db at a throwaway SQLite file for the duration of
    one test, so tests never touch the real urbanpulse.db."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # init_db() must be able to create it fresh
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield db
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------
# _haversine_m
# ---------------------------------------------------------------------

def test_haversine_zero_distance_for_identical_points():
    assert db._haversine_m(24.8607, 67.0011, 24.8607, 67.0011) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_within_tolerance():
    # Two points ~1 degree of latitude apart (~111.19 km at the equator's
    # meridian), a widely-cited reference distance for this formula.
    dist_m = db._haversine_m(0.0, 0.0, 1.0, 0.0)
    assert dist_m == pytest.approx(111195, rel=0.01)


def test_haversine_is_symmetric():
    a_to_b = db._haversine_m(24.8607, 67.0011, 24.8700, 67.0100)
    b_to_a = db._haversine_m(24.8700, 67.0100, 24.8607, 67.0011)
    assert a_to_b == pytest.approx(b_to_a, abs=1e-9)


# ---------------------------------------------------------------------
# save_map_pin — replace-within-radius rule
# ---------------------------------------------------------------------

KARACHI = {"latitude": 24.8607, "longitude": 67.0011}


def _pin(lat, lon, **overrides):
    pin = {"loc": {"latitude": lat, "longitude": lon}, "emotion": "Calm", "desc": "test pin"}
    pin.update(overrides)
    return pin


def test_new_pin_far_from_any_existing_is_inserted_not_replaced(temp_db):
    temp_db.save_map_pin(_pin(KARACHI["latitude"], KARACHI["longitude"], id="pin-1"))
    # ~11km away — far outside the 80m replace radius.
    temp_db.save_map_pin(_pin(KARACHI["latitude"] + 0.1, KARACHI["longitude"], id="pin-2"))

    pins = temp_db.get_map_pins()
    assert len(pins) == 2
    assert {p["id"] for p in pins} == {"pin-1", "pin-2"}


def test_pin_within_replace_radius_replaces_the_existing_one(temp_db):
    temp_db.save_map_pin(_pin(KARACHI["latitude"], KARACHI["longitude"], id="pin-old", desc="original"))
    # ~5m north of the original point — well within REPLACE_RADIUS_M (80m).
    nearby_lat = KARACHI["latitude"] + (5 / 111_195)
    temp_db.save_map_pin(_pin(nearby_lat, KARACHI["longitude"], id="pin-new", desc="replacement"))

    pins = temp_db.get_map_pins()
    assert len(pins) == 1
    assert pins[0]["id"] == "pin-new"
    assert pins[0]["desc"] == "replacement"


def test_pin_exactly_at_replace_radius_boundary_is_replaced(temp_db):
    # A point placed as close as reasonably achievable to exactly
    # REPLACE_RADIUS_M away should still trigger the <= comparison in
    # save_map_pin (boundary is inclusive).
    from models.db import REPLACE_RADIUS_M

    offset_deg = REPLACE_RADIUS_M / 111_195  # metres -> approx degrees latitude
    temp_db.save_map_pin(_pin(KARACHI["latitude"], KARACHI["longitude"], id="pin-old"))
    temp_db.save_map_pin(_pin(KARACHI["latitude"] + offset_deg, KARACHI["longitude"], id="pin-boundary"))

    pins = temp_db.get_map_pins()
    assert len(pins) == 1
    assert pins[0]["id"] == "pin-boundary"


def test_sound_hunt_completion_is_idempotent(temp_db):
    temp_db.mark_challenge_complete("rain")
    temp_db.mark_challenge_complete("rain")  # completing it again must not error or duplicate

    completed = temp_db.get_completed_challenges()
    assert completed.count("rain") == 1
