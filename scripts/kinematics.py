"""
KITE Phase 0 — per-track kinematic feature extraction.

Pure NumPy over a small ring buffer of track samples. No model, no video,
no OpenCV — unit-testable on synthetic trajectories.

Usage:
    store = TrackHistoryStore()
    feats = store.update(tracks, ts, frame_size=(w, h))   # every frame
    # feats: {track_id: KinematicFeatures}
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# Defaults (tunable). Window ~1 s at 30 FPS; min_len gates feature validity.
WINDOW = 30
MIN_LEN = 8
PERIODICITY_MIN_LEN = 12       # wingbeat needs more history than the rest
TRACK_TTL_S = 3.0              # evict a track this long after last update

# Per-step heading-change bands (radians) for turn_geometry.
SHARP_TURN = math.pi / 6       # > 30 deg/step = discrete geometric turn
SMOOTH_TURN = math.pi / 36     # 5..30 deg/step = smooth banking curve

# hover: speed below this fraction of frame diagonal per second,
# while the centroid stays inside a small bounding region.
HOVER_SPEED_FRAC = 0.02
HOVER_REGION_FRAC = 0.05


class Sample(NamedTuple):
    t: float        # seconds (perf_counter or media time)
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass(frozen=True)
class KinematicFeatures:
    n_samples: int
    duration_s: float
    # core motion
    straightness: float          # net displacement / path length, [0, 1]
    speed_mean: float            # px/s
    speed_std: float             # px/s
    heading_change_rate: float   # mean |dtheta| per second (rad/s)
    accel_mag: float             # mean |dv/dt| (px/s^2)
    jerk_mag: float              # mean |da/dt| (px/s^3)
    hover_score: float           # fraction of window spent hovering, [0, 1]
    turn_sharpness: float        # sharp turns / all turns, [0, 1]
    # periodicity (wingbeat signature) — may be unavailable on short tracks
    periodicity_available: bool
    vertical_periodicity_hz: float   # dominant freq of detrended cy, 0 if flat
    vertical_periodicity_power: float  # normalized peak power, [0, 1]
    aspect_oscillation: float    # std of bbox aspect ratio over window
    # normalized variants (by frame diagonal) so thresholds transfer
    speed_mean_norm: float
    speed_std_norm: float


def _headings(vx: np.ndarray, vy: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Heading angle per step, only where the target is actually moving —
    heading of a hovering point is noise."""
    moving = speed > 1e-3
    return np.arctan2(vy[moving], vx[moving])


def _angle_diff(a: np.ndarray) -> np.ndarray:
    """Successive heading differences wrapped to [-pi, pi]."""
    d = np.diff(a)
    return (d + math.pi) % (2 * math.pi) - math.pi


def _dominant_frequency(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Dominant frequency (Hz) and normalized power of a detrended signal.

    Resamples to a uniform grid (tracker timestamps are never uniform),
    removes the linear trend, then takes the peak of the rFFT power
    spectrum excluding DC. Power is normalized by total spectral power.
    """
    duration = t[-1] - t[0]
    if duration <= 0:
        return 0.0, 0.0
    n = max(32, len(t))
    tu = np.linspace(t[0], t[-1], n)
    yu = np.interp(tu, t, y)
    # detrend (linear) so slow drift doesn't read as a low frequency
    coef = np.polyfit(tu, yu, 1)
    yd = yu - np.polyval(coef, tu)
    if np.allclose(yd, 0, atol=1e-9):
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(yd)) ** 2
    freqs = np.fft.rfftfreq(n, d=duration / (n - 1))
    spec[0] = 0.0                          # kill DC
    total = spec.sum()
    if total <= 0:
        return 0.0, 0.0
    k = int(np.argmax(spec))
    return float(freqs[k]), float(spec[k] / total)


def extract_features(samples: list[Sample],
                     frame_size: tuple[float, float] | None = None,
                     min_len: int = MIN_LEN) -> KinematicFeatures | None:
    """Compute the kinematic feature vector for one track's history.

    Returns None when the track is too short (< min_len samples) —
    callers fall back to appearance-only classification.
    """
    if len(samples) < min_len:
        return None

    t = np.array([s.t for s in samples], dtype=np.float64)
    cx = np.array([s.cx for s in samples], dtype=np.float64)
    cy = np.array([s.cy for s in samples], dtype=np.float64)
    w = np.array([s.w for s in samples], dtype=np.float64)
    h = np.array([s.h for s in samples], dtype=np.float64)

    dt = np.diff(t)
    if np.any(dt <= 0):                    # defensive: non-monotonic time
        keep = np.concatenate(([True], dt > 0))
        t, cx, cy, w, h = t[keep], cx[keep], cy[keep], w[keep], h[keep]
        if len(t) < min_len:
            return None
        dt = np.diff(t)

    duration = float(t[-1] - t[0])
    diag = math.hypot(*frame_size) if frame_size else 1.0

    dx, dy = np.diff(cx), np.diff(cy)
    step = np.hypot(dx, dy)
    path_len = float(step.sum())
    net_disp = math.hypot(float(cx[-1] - cx[0]), float(cy[-1] - cy[0]))
    straightness = net_disp / path_len if path_len > 1e-9 else 1.0

    vx, vy = dx / dt, dy / dt
    speed = np.hypot(vx, vy)
    speed_mean = float(speed.mean())
    speed_std = float(speed.std())

    # heading change per unit time
    headings = _headings(vx, vy, speed)
    if len(headings) >= 2 and duration > 0:
        dtheta = _angle_diff(headings)
        heading_change_rate = float(np.abs(dtheta).sum() / duration)
        n_sharp = int((np.abs(dtheta) > SHARP_TURN).sum())
        n_turn = int((np.abs(dtheta) > SMOOTH_TURN).sum())
        turn_sharpness = n_sharp / n_turn if n_turn else 0.0
    else:
        heading_change_rate = 0.0
        turn_sharpness = 0.0

    # acceleration / jerk magnitudes
    if len(vx) >= 2:
        tm = (t[:-1] + t[1:]) / 2          # velocity sample midpoints
        dtv = np.diff(tm)
        ax, ay = np.diff(vx) / dtv, np.diff(vy) / dtv
        accel = np.hypot(ax, ay)
        accel_mag = float(accel.mean())
        jerk_mag = float(np.abs(np.diff(accel) / np.diff((tm[:-1] + tm[1:]) / 2)).mean()) \
            if len(accel) >= 2 else 0.0
    else:
        accel_mag = 0.0
        jerk_mag = 0.0

    # hover: slow AND spatially bounded. Speed is measured over a coarse
    # multi-frame baseline — per-frame centroid jitter from detection noise
    # (easily +-2 px/frame = 60 px/s) would otherwise swamp a true hover.
    hover_speed_thresh = HOVER_SPEED_FRAC * diag if frame_size else 15.0
    region = max(float(cx.max() - cx.min()), float(cy.max() - cy.min()))
    region_thresh = HOVER_REGION_FRAC * diag if frame_size else 40.0
    if region <= region_thresh:
        k = min(5, len(t) - 1)
        disp = np.hypot(cx[k:] - cx[:-k], cy[k:] - cy[:-k])
        base_dt = t[k:] - t[:-k]
        coarse_speed = disp / np.maximum(base_dt, 1e-9)
        hover_score = float((coarse_speed < hover_speed_thresh).mean())
    else:
        hover_score = 0.0

    # wingbeat periodicity — needs more history
    if len(t) >= PERIODICITY_MIN_LEN and duration >= 0.4:
        periodicity_available = True
        vp_hz, vp_power = _dominant_frequency(t, cy)
        aspect = np.divide(w, h, out=np.ones_like(w), where=h > 1e-9)
        aspect_osc = float(aspect.std())
    else:
        periodicity_available = False
        vp_hz, vp_power, aspect_osc = 0.0, 0.0, 0.0

    return KinematicFeatures(
        n_samples=len(t),
        duration_s=duration,
        straightness=float(straightness),
        speed_mean=speed_mean,
        speed_std=speed_std,
        heading_change_rate=heading_change_rate,
        accel_mag=accel_mag,
        jerk_mag=jerk_mag,
        hover_score=hover_score,
        turn_sharpness=float(turn_sharpness),
        periodicity_available=periodicity_available,
        vertical_periodicity_hz=vp_hz,
        vertical_periodicity_power=vp_power,
        aspect_oscillation=aspect_osc,
        speed_mean_norm=speed_mean / diag,
        speed_std_norm=speed_std / diag,
    )


class TrackHistory:
    """Ring buffer of Samples for one track id.

    Features over a ~1 s window change slowly, so they are recomputed only
    every FEATURE_STRIDE new samples and cached between — this is what keeps
    the intel stage cheap with dozens of concurrent tracks.
    """

    FEATURE_STRIDE = 3

    def __init__(self, window: int = WINDOW):
        self.samples: deque[Sample] = deque(maxlen=window)
        self.last_seen: float = 0.0
        self._cached: KinematicFeatures | None = None
        self._since_compute = 0

    def add(self, s: Sample) -> None:
        self.samples.append(s)
        self.last_seen = s.t
        self._since_compute += 1

    def features(self, frame_size=None, min_len: int = MIN_LEN):
        if self._cached is None or self._since_compute >= self.FEATURE_STRIDE:
            self._cached = extract_features(list(self.samples), frame_size,
                                            min_len)
            self._since_compute = 0
        return self._cached


class TrackHistoryStore:
    """All live track histories; call update() once per frame."""

    def __init__(self, window: int = WINDOW, ttl_s: float = TRACK_TTL_S):
        self.window = window
        self.ttl_s = ttl_s
        self.tracks: dict[int, TrackHistory] = {}

    def update(self, tracks: list[dict], ts: float,
               frame_size: tuple[float, float] | None = None,
               ) -> dict[int, KinematicFeatures | None]:
        """Ingest one frame's tracked detections.

        tracks: [{id, bbox: [x1,y1,x2,y2], conf, ...}] — the exact dicts
        pipeline_demo.py's tracking stage already produces.
        Returns {track_id: KinematicFeatures | None (too short)}.
        """
        out: dict[int, KinematicFeatures | None] = {}
        for tr in tracks:
            tid = tr["id"]
            x1, y1, x2, y2 = tr["bbox"]
            s = Sample(t=ts, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                       w=x2 - x1, h=y2 - y1, conf=tr.get("conf", 0.0))
            hist = self.tracks.get(tid)
            if hist is None:
                hist = self.tracks[tid] = TrackHistory(self.window)
            hist.add(s)
            out[tid] = hist.features(frame_size)
        # evict stale tracks so memory stays bounded on long streams
        stale = [tid for tid, hist in self.tracks.items()
                 if ts - hist.last_seen > self.ttl_s]
        for tid in stale:
            del self.tracks[tid]
        return out

    def trail(self, tid: int, n: int = WINDOW) -> list[tuple[float, float]]:
        hist = self.tracks.get(tid)
        if hist is None:
            return []
        return [(s.cx, s.cy) for s in list(hist.samples)[-n:]]
