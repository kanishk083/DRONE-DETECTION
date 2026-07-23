"""
KITE Phase 1 — appearance + motion fusion classifier.

Rule-based scorer: combines YOLO's appearance confidence with a motion
score derived from kinematic features. Reweights — never blindly
overrides — the appearance call, and returns a human-readable reason
string for the dashboard.

    fused = fuse(cls="bird", conf=0.62, feats=<KinematicFeatures>)
    fused.label, fused.confidence, fused.reason
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kinematics import KinematicFeatures

# Motion-score weights (tuned on synthetic + test clips; positive = drone-like).
W_STRAIGHT = 1.4        # straight-line cruise
W_HOVER = 1.8           # sustained precise hover — strongest drone signature
W_LOW_SPEED_STD = 0.8   # constant velocity
W_TURN_SHARP = 0.6      # discrete geometric turns
W_WINGBEAT = 2.2        # periodic vertical oscillation — strongest bird signature
W_ASPECT = 1.2          # silhouette oscillation from flapping

# Wingbeat band: real birds flap ~2-8 Hz on camera. Power below this
# threshold is treated as noise, not signal.
WINGBEAT_MIN_HZ = 1.5
WINGBEAT_MAX_HZ = 10.0
WINGBEAT_MIN_POWER = 0.25

# Fusion: p = sigmoid(a*appearance_logit + b*motion + c)
FUSE_A = 1.0
FUSE_B = 0.9
FUSE_C = 0.0

# Decision threshold on fused P(drone), chosen from the measured sweep in
# KINEMATICS_RESULTS.md: max balanced accuracy subject to drone recall
# >= 80% (a missed drone costs more than a false alarm, so recall floors
# the sweep rather than symmetric 0.5).
DRONE_THRESHOLD = 0.55


@dataclass(frozen=True)
class FusedResult:
    label: str              # corrected class
    confidence: float       # calibrated fused P(label)
    appearance_conf: float  # raw YOLO confidence for its own call
    motion_score: float     # positive = drone-like, negative = bird-like
    reason: str             # short human-readable explanation
    flagged: bool           # appearance and motion disagree strongly


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def motion_score(f: KinematicFeatures) -> tuple[float, list[str]]:
    """Signed drone-vs-bird motion score plus the evidence behind it.

    Positive = moves like a drone; negative = moves like a bird.
    """
    score = 0.0
    reasons: list[str] = []

    if f.straightness > 0.9:
        score += W_STRAIGHT * (f.straightness - 0.9) / 0.1
        reasons.append("straight-line cruise")

    if f.hover_score > 0.5:
        score += W_HOVER * f.hover_score
        reasons.append("sustained hover")

    # constant velocity (low relative speed variation while moving)
    if f.speed_mean > 1e-3:
        rel_std = f.speed_std / f.speed_mean
        if rel_std < 0.25 and f.hover_score < 0.5:
            score += W_LOW_SPEED_STD * (0.25 - rel_std) / 0.25
            reasons.append("constant velocity")

    if f.turn_sharpness > 0.5:
        score += W_TURN_SHARP * f.turn_sharpness
        reasons.append("geometric turns")

    if f.periodicity_available:
        wingbeat = (WINGBEAT_MIN_HZ <= f.vertical_periodicity_hz <= WINGBEAT_MAX_HZ
                    and f.vertical_periodicity_power >= WINGBEAT_MIN_POWER)
        if wingbeat:
            score -= W_WINGBEAT * f.vertical_periodicity_power
            reasons.append(f"wingbeat {f.vertical_periodicity_hz:.1f} Hz")
        elif f.vertical_periodicity_power < WINGBEAT_MIN_POWER:
            reasons.append("no wingbeat")

        if f.aspect_oscillation > 0.08:
            score -= W_ASPECT * min(f.aspect_oscillation / 0.2, 1.0)
            reasons.append("silhouette oscillation")

    return score, reasons


def fuse(cls: str, conf: float,
         feats: KinematicFeatures | None) -> FusedResult:
    """Fuse YOLO's appearance call with the track's motion signature.

    feats=None (track too short) falls back to appearance unchanged.
    """
    if feats is None:
        return FusedResult(label=cls, confidence=conf, appearance_conf=conf,
                           motion_score=0.0, reason="insufficient history",
                           flagged=False)

    m_score, reasons = motion_score(feats)

    # appearance as P(drone): YOLO conf is for its own predicted class
    p_drone_app = conf if cls == "drone" else 1.0 - conf
    p_drone = _sigmoid(FUSE_A * _logit(p_drone_app) + FUSE_B * m_score + FUSE_C)

    label = "drone" if p_drone >= DRONE_THRESHOLD else "bird"
    confidence = p_drone if label == "drone" else 1.0 - p_drone

    # flag strong appearance-vs-motion disagreement for review
    flagged = (label != cls and conf > 0.5) or \
              (cls == "bird" and m_score > 2.0) or \
              (cls == "drone" and m_score < -2.0)

    reason = ", ".join(reasons) if reasons else "no strong motion signal"
    return FusedResult(label=label, confidence=confidence,
                       appearance_conf=conf, motion_score=m_score,
                       reason=reason, flagged=flagged)
