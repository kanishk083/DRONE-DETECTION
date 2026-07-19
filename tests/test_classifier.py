"""KITE Phase 1 tests — fusion classifier on synthetic tracks."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from kinematics import Sample, extract_features  # noqa: E402
from classifier import fuse, motion_score  # noqa: E402

FRAME = (768.0, 432.0)
FPS = 30.0


def make(fn, n=30):
    return [Sample(i / FPS, *fn(i / FPS), 0.8) for i in range(n)]


def cruise(t):
    return 100 + 120 * t, 200 + 30 * t, 40, 30


def flap(t):
    cx = 100 + 60 * t + 15 * math.sin(2 * math.pi * 0.7 * t)
    cy = 200 + 10 * t + 8 * math.sin(2 * math.pi * 4.0 * t)
    w = 30 + 8 * math.sin(2 * math.pi * 4.0 * t)
    return cx, cy, w, 20


def hover(t):
    j = abs(math.sin(t * 50))
    return 300 + j * 1.5, 150 + j * 1.2, 40, 30


def test_motion_score_signs():
    drone_score, _ = motion_score(extract_features(make(cruise), FRAME))
    bird_score, _ = motion_score(extract_features(make(flap), FRAME))
    assert drone_score > 0
    assert bird_score < drone_score


def test_fusion_boosts_agreeing_drone():
    f = extract_features(make(cruise), FRAME)
    r = fuse("drone", 0.55, f)
    assert r.label == "drone"
    assert r.confidence > 0.55          # motion agrees -> more confident
    assert "cruise" in r.reason or "constant velocity" in r.reason


def test_fusion_corrects_bird_that_cruises_and_hovers():
    f = extract_features(make(hover), FRAME)
    r = fuse("bird", 0.52, f)           # weak appearance bird, hovers like drone
    assert r.label == "drone"
    assert r.flagged


def test_fusion_respects_strong_bird_with_wingbeat():
    f = extract_features(make(flap), FRAME)
    r = fuse("bird", 0.85, f)
    assert r.label == "bird"
    assert r.confidence >= 0.85 - 1e-9  # wingbeat reinforces bird


def test_short_track_falls_back_to_appearance():
    r = fuse("bird", 0.7, None)
    assert r.label == "bird"
    assert r.confidence == 0.7
    assert r.reason == "insufficient history"
    assert not r.flagged


def test_high_conf_drone_not_flipped_by_ambiguous_motion():
    # weak motion evidence must not override a confident appearance call
    f = extract_features(make(cruise, n=12), FRAME)
    r = fuse("drone", 0.95, f)
    assert r.label == "drone"
