"""
KITE Phase 0 tests — synthetic trajectories, no video or model needed.

Acceptance (from the KITE plan):
  - straight line scores straightness > 0.95
  - sine-wave "flapping" reports non-zero vertical_periodicity_hz
  - hover scores hover_score > 0.8
  - short tracks return None (appearance-only fallback)
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from kinematics import (  # noqa: E402
    MIN_LEN,
    Sample,
    TrackHistoryStore,
    extract_features,
)

FRAME = (768.0, 432.0)
FPS = 30.0


def make_track(fn, n=30, fps=FPS):
    """Build samples from fn(t) -> (cx, cy, w, h)."""
    out = []
    for i in range(n):
        t = i / fps
        cx, cy, w, h = fn(t)
        out.append(Sample(t=t, cx=cx, cy=cy, w=w, h=h, conf=0.8))
    return out


# --- trajectory generators --------------------------------------------------

def straight_cruise(t):
    """Drone cruising: constant velocity, fixed box."""
    return 100 + 120 * t, 200 + 30 * t, 40, 30


def hover(t):
    """Drone hovering: sub-pixel jitter around a fixed point."""
    rng = abs(math.sin(t * 50))  # deterministic tiny jitter
    return 300 + rng * 1.5, 150 + rng * 1.2, 40, 30


def flapping_bird(t):
    """Bird: wandering path + few-Hz vertical wingbeat oscillation
    + aspect ratio oscillation as the wings change silhouette."""
    cx = 100 + 60 * t + 15 * math.sin(2 * math.pi * 0.7 * t)
    cy = 200 + 10 * t + 8 * math.sin(2 * math.pi * 4.0 * t)   # 4 Hz wingbeat
    w = 30 + 8 * math.sin(2 * math.pi * 4.0 * t)
    return cx, cy, w, 20


def sharp_turn(t):
    """Waypoint flight: straight, 90-degree corner, straight."""
    if t < 0.5:
        return 100 + 200 * t, 200.0, 40, 30
    return 200.0, 200 + 200 * (t - 0.5), 40, 30


# --- acceptance tests -------------------------------------------------------

def test_straight_line_high_straightness():
    f = extract_features(make_track(straight_cruise), FRAME)
    assert f is not None
    assert f.straightness > 0.95
    assert f.speed_std < 0.2 * max(f.speed_mean, 1.0)  # near-constant velocity


def test_hover_score_high():
    f = extract_features(make_track(hover, n=30), FRAME)
    assert f is not None
    assert f.hover_score > 0.8
    assert f.speed_mean_norm < 0.05


def test_flapping_bird_periodicity():
    f = extract_features(make_track(flapping_bird, n=30), FRAME)
    assert f is not None
    assert f.periodicity_available
    assert f.vertical_periodicity_hz > 0
    # 4 Hz wingbeat should be recovered within ~1 Hz
    assert abs(f.vertical_periodicity_hz - 4.0) < 1.0
    assert f.vertical_periodicity_power > 0.3
    assert f.aspect_oscillation > 0.05


def test_drone_flat_periodicity_vs_bird():
    drone = extract_features(make_track(straight_cruise), FRAME)
    bird = extract_features(make_track(flapping_bird), FRAME)
    assert drone.vertical_periodicity_power < bird.vertical_periodicity_power
    assert drone.aspect_oscillation < bird.aspect_oscillation


def test_sharp_turn_geometry():
    f = extract_features(make_track(sharp_turn), FRAME)
    assert f is not None
    assert f.turn_sharpness > 0.5          # one discrete corner dominates
    assert f.straightness < 0.9            # corner breaks straightness


def test_short_track_returns_none():
    assert extract_features(make_track(straight_cruise, n=MIN_LEN - 1), FRAME) is None


def test_short_track_no_periodicity_flag():
    f = extract_features(make_track(straight_cruise, n=9), FRAME)
    assert f is not None
    assert not f.periodicity_available


# --- store behaviour --------------------------------------------------------

def _tracks_dict(tid, cx, cy):
    return [{"id": tid, "class": "drone", "conf": 0.8,
             "bbox": [cx - 20, cy - 15, cx + 20, cy + 15]}]


def test_store_accumulates_and_extracts():
    store = TrackHistoryStore()
    feats = {}
    for i in range(30):
        t = i / FPS
        feats = store.update(_tracks_dict(7, 100 + 120 * t, 200), t, FRAME)
    assert 7 in feats
    assert feats[7] is not None
    assert feats[7].straightness > 0.95


def test_store_returns_none_before_min_len():
    store = TrackHistoryStore()
    feats = store.update(_tracks_dict(1, 100, 100), 0.0, FRAME)
    assert feats[1] is None


def test_store_evicts_stale_tracks():
    store = TrackHistoryStore(ttl_s=1.0)
    store.update(_tracks_dict(1, 100, 100), 0.0, FRAME)
    store.update(_tracks_dict(2, 300, 300), 5.0, FRAME)  # track 1 now stale
    assert 1 not in store.tracks
    assert 2 in store.tracks


def test_store_trail():
    store = TrackHistoryStore()
    for i in range(10):
        store.update(_tracks_dict(3, 100 + i * 10, 200), i / FPS, FRAME)
    trail = store.trail(3)
    assert len(trail) == 10
    assert trail[-1][0] == pytest.approx(190)


def test_non_monotonic_timestamps_survive():
    samples = make_track(straight_cruise, n=20)
    samples[10] = samples[10]._replace(t=samples[9].t)  # duplicate timestamp
    f = extract_features(samples, FRAME)
    assert f is not None  # dropped the bad sample, still extracted


def test_window_ring_buffer_bounded():
    store = TrackHistoryStore(window=30)
    for i in range(100):
        store.update(_tracks_dict(5, 100 + i, 200), i / FPS, FRAME)
    assert len(store.tracks[5].samples) == 30
