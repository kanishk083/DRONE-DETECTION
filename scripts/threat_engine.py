"""
KITE Phase 2 — Kalman trajectory prediction, keep-out zones, threat
scoring, and debounced tactical events.

Pure NumPy. ByteTrack's internal Kalman state is not cleanly exposed, so
each track gets a small dedicated constant-velocity filter on its
centroid — readable, and it yields the predicted path + uncertainty the
dashboard draws.

    engine = ThreatEngine(zones=[{"name": "pad", "points": [[x, y], ...]}])
    events = engine.update(tracks, ts, frame_size)   # mutates tracks in place
"""

from __future__ import annotations

import math

import numpy as np

PREDICT_STEPS = 15          # ~0.5 s at 30 FPS
EVENT_DEBOUNCE_S = 2.0      # min gap between repeats of the same event
LOITER_S = 3.0              # sustained hover this long near a zone => LOITERING
TRACK_TTL_S = 3.0

SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ---------------------------------------------------------------------------
# Constant-velocity Kalman filter over the centroid
# ---------------------------------------------------------------------------

class CVKalman:
    """State [cx, cy, vx, vy]; observe [cx, cy]."""

    def __init__(self, cx: float, cy: float,
                 q: float = 40.0, r: float = 8.0):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([r, r, 500.0, 500.0])
        self.q = q                       # process noise (accel spectral density)
        self.R = np.eye(2) * r ** 2
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.last_t: float | None = None

    def _F_Q(self, dt: float):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # white-accel process noise
        q = self.q ** 2
        dt2, dt3 = dt * dt, dt * dt * dt
        Q = q * np.array([
            [dt3 / 3, 0, dt2 / 2, 0],
            [0, dt3 / 3, 0, dt2 / 2],
            [dt2 / 2, 0, dt, 0],
            [0, dt2 / 2, 0, dt],
        ])
        return F, Q

    def step(self, cx: float, cy: float, t: float) -> None:
        if self.last_t is None:
            self.x[:2] = (cx, cy)
            self.last_t = t
            return
        dt = max(t - self.last_t, 1e-6)
        self.last_t = t
        F, Q = self._F_Q(dt)
        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # update
        z = np.array([cx, cy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict_path(self, steps: int = PREDICT_STEPS,
                     dt: float = 1 / 30) -> list[tuple[float, float]]:
        cx, cy, vx, vy = self.x
        return [(cx + vx * dt * k, cy + vy * dt * k)
                for k in range(1, steps + 1)]

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


# ---------------------------------------------------------------------------
# Zone geometry
# ---------------------------------------------------------------------------

def point_in_polygon(px: float, py: float, poly: list) -> bool:
    """Ray-casting point-in-polygon."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > py) != (yj > py) and \
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def dist_to_polygon(px: float, py: float, poly: list) -> float:
    """Distance from a point to the polygon boundary (0 if inside)."""
    if point_in_polygon(px, py, poly):
        return 0.0
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg2))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


# ---------------------------------------------------------------------------
# Threat engine
# ---------------------------------------------------------------------------

class ThreatEngine:
    def __init__(self, zones: list | None = None):
        self.zones = zones or []          # [{"name", "points": [[x,y],...]}]
        self.kalman: dict[int, CVKalman] = {}
        self.last_seen: dict[int, float] = {}
        self.known_tracks: set[int] = set()
        self.classified_drone: set[int] = set()
        self.in_zone: dict[int, bool] = {}
        self.hover_since: dict[int, float] = {}
        self._last_event: dict[tuple, float] = {}   # (type, tid) -> ts

    def set_zones(self, zones: list) -> None:
        self.zones = zones or []

    # -- events ------------------------------------------------------------
    def _emit(self, events: list, etype: str, tid: int, severity: str,
              ts: float, **extra) -> None:
        key = (etype, tid)
        if ts - self._last_event.get(key, -1e9) < EVENT_DEBOUNCE_S:
            return
        self._last_event[key] = ts
        events.append({"type": etype, "track_id": tid,
                       "severity": severity, "ts": ts, **extra})

    # -- main per-frame update ----------------------------------------------
    def update(self, tracks: list[dict], ts: float,
               frame_size: tuple[float, float]) -> list[dict]:
        """Enrich each track dict in place with 'predicted' and 'threat';
        return this frame's events."""
        events: list[dict] = []
        diag = math.hypot(*frame_size)
        live_ids = set()

        for tr in tracks:
            tid = tr["id"]
            live_ids.add(tid)
            x1, y1, x2, y2 = tr["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cls = tr.get("fused_class", tr["class"])
            conf = tr.get("fused_conf", tr["conf"])
            feats = tr.get("kinematics")

            # Kalman: step + predicted path
            kf = self.kalman.get(tid)
            if kf is None:
                kf = self.kalman[tid] = CVKalman(cx, cy)
            kf.step(cx, cy, ts)
            predicted = kf.predict_path()
            tr["predicted"] = predicted
            self.last_seen[tid] = ts

            # lifecycle events
            if tid not in self.known_tracks:
                self.known_tracks.add(tid)
                self._emit(events, "NEW_TRACK", tid, "LOW", ts, cls=cls)
            if cls == "drone" and conf > 0.6 and tid not in self.classified_drone:
                self.classified_drone.add(tid)
                self._emit(events, "CLASSIFIED_DRONE", tid, "MEDIUM", ts,
                           conf=round(conf, 2))

            # threat scoring (drones only; birds always NONE)
            if cls != "drone":
                tr["threat"] = {"score": 0, "level": "NONE",
                                "zone_inbound": False, "eta_s": None}
                continue

            score = 25.0 * conf            # base: how sure we are it's a drone
            zone_inbound = False
            eta_s = None
            inside = False

            if self.zones:
                d_now = min(dist_to_polygon(cx, cy, z["points"])
                            for z in self.zones)
                inside = d_now == 0.0
                # proximity: within 25% of frame diagonal ramps 0..30
                score += max(0.0, 1.0 - d_now / (0.25 * diag)) * 30.0
                # inbound: predicted path enters (or approaches) a zone
                if not inside:
                    for k, (px, py) in enumerate(predicted):
                        if any(point_in_polygon(px, py, z["points"])
                               for z in self.zones):
                            zone_inbound = True
                            vx, vy = kf.velocity
                            v = math.hypot(vx, vy)
                            eta_s = d_now / v if v > 1.0 else None
                            break
                    if zone_inbound:
                        score += 25.0
                if inside:
                    score += 40.0
                    if not self.in_zone.get(tid):
                        self._emit(events, "ZONE_BREACH", tid, "CRITICAL", ts)
                elif zone_inbound and not self.in_zone.get(tid):
                    self._emit(events, "ZONE_INBOUND", tid, "HIGH", ts,
                               eta_s=round(eta_s, 1) if eta_s else None)
                self.in_zone[tid] = inside

            # hover / loitering
            if feats is not None and feats.hover_score > 0.6:
                score += 10.0
                if tid not in self.hover_since:
                    self.hover_since[tid] = ts
                elif ts - self.hover_since[tid] > LOITER_S:
                    self._emit(events, "LOITERING", tid, "MEDIUM", ts)
            else:
                self.hover_since.pop(tid, None)

            # altitude proxy: high in frame + small box reads as distant/high
            score += (1.0 - cy / frame_size[1]) * 5.0

            score = max(0.0, min(100.0, score))
            level = ("CRITICAL" if inside or score >= 85 else
                     "HIGH" if score >= 65 else
                     "MEDIUM" if score >= 45 else
                     "LOW" if score >= 25 else "NONE")
            tr["threat"] = {"score": int(round(score)), "level": level,
                            "zone_inbound": zone_inbound,
                            "eta_s": round(eta_s, 1) if eta_s else None}

        # lost tracks
        for tid in list(self.last_seen):
            if tid not in live_ids and ts - self.last_seen[tid] > TRACK_TTL_S:
                self._emit(events, "TRACK_LOST", tid, "LOW", ts)
                for d in (self.kalman, self.last_seen, self.in_zone,
                          self.hover_since):
                    d.pop(tid, None)
                self.known_tracks.discard(tid)
                self.classified_drone.discard(tid)

        return events
