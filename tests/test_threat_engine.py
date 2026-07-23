"""KITE Phase 2 tests — Kalman, zones, threat scoring, events, debounce."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from threat_engine import (  # noqa: E402
    CVKalman,
    ThreatEngine,
    dist_to_polygon,
    point_in_polygon,
)

FRAME = (768.0, 432.0)
FPS = 30.0
SQUARE = [[500, 100], [700, 100], [700, 300], [500, 300]]


def track(tid, cx, cy, cls="drone", conf=0.9):
    return {"id": tid, "class": cls, "conf": conf,
            "fused_class": cls, "fused_conf": conf,
            "bbox": [cx - 20, cy - 15, cx + 20, cy + 15], "kinematics": None}


def run(engine, positions, tid=1, cls="drone", conf=0.9):
    """Feed a sequence of centroids; collect all events + final tracks."""
    all_events, tracks = [], []
    for i, (cx, cy) in enumerate(positions):
        tracks = [track(tid, cx, cy, cls, conf)]
        all_events += engine.update(tracks, i / FPS, FRAME)
    return all_events, tracks


# --- geometry ---------------------------------------------------------------

def test_point_in_polygon():
    assert point_in_polygon(600, 200, SQUARE)
    assert not point_in_polygon(100, 100, SQUARE)


def test_dist_to_polygon():
    assert dist_to_polygon(600, 200, SQUARE) == 0.0
    assert dist_to_polygon(400, 200, SQUARE) == pytest.approx(100.0)


# --- kalman -----------------------------------------------------------------

def test_kalman_beats_constant_position():
    kf = CVKalman(0, 0)
    for i in range(20):                       # constant velocity 120 px/s in x
        kf.step(120 * i / FPS, 200.0, i / FPS)
    path = kf.predict_path(steps=15, dt=1 / FPS)
    true_x = 120 * (19 + 15) / FPS
    pred_err = abs(path[-1][0] - true_x)
    const_pos_err = abs(120 * 19 / FPS - true_x)
    assert pred_err < const_pos_err           # must beat constant-position
    assert pred_err < 15.0                    # and be genuinely close (px)


# --- events -----------------------------------------------------------------

def test_zone_breach_event():
    engine = ThreatEngine([{"name": "pad", "points": SQUARE}])
    # fly straight into the square
    events, tracks = run(engine, [(300 + 10 * i, 200) for i in range(30)])
    types = [e["type"] for e in events]
    assert "NEW_TRACK" in types
    assert "ZONE_BREACH" in types
    assert tracks[0]["threat"]["level"] == "CRITICAL"


def test_zone_inbound_before_breach():
    engine = ThreatEngine([{"name": "pad", "points": SQUARE}])
    events, _ = run(engine, [(200 + 10 * i, 200) for i in range(25)])
    types = [e["type"] for e in events]
    assert "ZONE_INBOUND" in types
    assert "ZONE_BREACH" not in types         # never actually entered


def test_bird_scores_none():
    engine = ThreatEngine([{"name": "pad", "points": SQUARE}])
    _, tracks = run(engine, [(300 + 15 * i, 200) for i in range(30)],
                    cls="bird")
    assert tracks[0]["threat"]["level"] == "NONE"
    assert tracks[0]["threat"]["score"] == 0


def test_event_debounce():
    engine = ThreatEngine([{"name": "pad", "points": SQUARE}])
    # sit exactly on the boundary-crossing spot frame after frame
    events = []
    for i in range(60):                       # 2 s inside the zone
        events += engine.update([track(1, 600, 200)], i / FPS, FRAME)
    breaches = [e for e in events if e["type"] == "ZONE_BREACH"]
    assert len(breaches) == 1                 # debounced, no spam


def test_predicted_path_attached():
    engine = ThreatEngine()
    _, tracks = run(engine, [(100 + 10 * i, 200) for i in range(10)])
    assert len(tracks[0]["predicted"]) == 15


def test_track_lost_event():
    engine = ThreatEngine()
    engine.update([track(1, 100, 100)], 0.0, FRAME)
    events = engine.update([track(2, 300, 300)], 5.0, FRAME)  # track 1 gone
    assert "TRACK_LOST" in [e["type"] for e in events]
    assert 1 not in engine.kalman


def test_classified_drone_once():
    engine = ThreatEngine()
    events, _ = run(engine, [(100 + 5 * i, 200) for i in range(90)])
    assert [e["type"] for e in events].count("CLASSIFIED_DRONE") == 1
