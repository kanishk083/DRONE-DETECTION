"""
Multi-threaded detection + tracking pipeline (3 threads + shared buffers).

Architecture (each stage runs concurrently on different frames):

  [Camera thread] --Frame Buffer--> [Detection thread] --Detection Buffer-->
  [Tracking thread] --Tracking Buffer--> main thread (draw / count FPS)

Usage:
  python scripts/pipeline_demo.py --source test_assets/drone_test.mp4
  python scripts/pipeline_demo.py --source 0                 # webcam
  python scripts/pipeline_demo.py --sequential               # baseline mode
  python scripts/pipeline_demo.py --model models/best_int8.onnx --imgsz 416
  python scripts/pipeline_demo.py --no-show                  # benchmark only
"""

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# BGR colors: red for drones (threat), sky-blue for birds — matches the web UI.
CLASS_COLORS = {"drone": (0, 0, 255), "bird": (235, 206, 135)}


# ---------------------------------------------------------------------------
# Shared-memory buffers (packets passed between threads)
# ---------------------------------------------------------------------------

@dataclass
class FramePacket:
    frame_id: int
    frame: np.ndarray
    ts: float                      # capture timestamp (for latency stats)


@dataclass
class DetPacket:
    frame_id: int
    frame: np.ndarray
    detections: sv.Detections
    ts: float
    infer_ms: float


@dataclass
class TrackPacket:
    frame_id: int
    frame: np.ndarray
    ts: float
    infer_ms: float
    tracks: list = field(default_factory=list)   # [{id, class, conf, bbox}]


@dataclass
class IntelPacket:
    """TrackPacket + KITE intelligence: fused classification, kinematics,
    trails (and, in Phase 2, threat + events)."""
    frame_id: int
    frame: np.ndarray
    ts: float
    infer_ms: float
    intel_ms: float
    tracks: list = field(default_factory=list)
    events: list = field(default_factory=list)


class LatestQueue(queue.Queue):
    """maxsize=1 queue that DROPS the oldest item instead of blocking.
    Used in live mode so the pipeline always works on the newest frame
    and never builds up lag behind the camera."""

    def __init__(self):
        super().__init__(maxsize=1)

    def put(self, item, block=False, timeout=None):
        try:
            self.get_nowait()          # throw away stale frame if present
        except queue.Empty:
            pass
        try:
            super().put_nowait(item)
        except queue.Full:
            pass


# Pushed through the buffers to signal "video finished". Must be a unique
# object (NOT None) because _pop() returns None when a queue is merely
# empty at that instant — confusing the two makes consumers quit early.
_SENTINEL = object()


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

class ThreadedPipeline:
    def __init__(self, model_path, source, imgsz=640, conf=0.3, live=None,
                 device=None, intel=False, zones=None):
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.source = source
        self.imgsz = imgsz
        self.conf = conf
        self.device = device            # e.g. "intel:gpu" for OpenVINO on iGPU
        # Webcam => live mode (drop frames to stay realtime).
        # Video file => process every frame (blocking queues, no drops).
        self.live = live if live is not None else isinstance(source, int)

        # KITE intelligence stage (4th thread): kinematics + fusion (+ threat)
        self.intel = intel
        self.zones = zones or []
        # File sources are processed off-realtime, so kinematic features must
        # use media time (frame_id / video fps), not the wall clock.
        self.video_fps = 30.0
        if not isinstance(source, int):
            probe = cv2.VideoCapture(source)
            fps = probe.get(cv2.CAP_PROP_FPS)
            probe.release()
            if fps and fps > 0:
                self.video_fps = fps

        make_q = LatestQueue if self.live else lambda: queue.Queue(maxsize=32)
        self.frame_buffer = make_q()       # camera  -> detection
        self.det_buffer = make_q()         # detect  -> tracking
        self.track_buffer = make_q()       # track   -> intel / main thread
        self.intel_buffer = make_q()       # intel   -> main thread

        self.tracker = sv.ByteTrack()
        self._stop = threading.Event()
        self.threads = []

        # warmup: first inference compiles/allocates and is 10x slower —
        # do it here so it doesn't pollute FPS numbers
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=imgsz, verbose=False, device=device)

    # --- Thread 1: camera/ingest ------------------------------------------
    def _camera_loop(self):
        cap = cv2.VideoCapture(self.source)
        frame_id = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:                       # end of video file
                    break
                pkt = FramePacket(frame_id, frame, time.perf_counter())
                if self.live:
                    self.frame_buffer.put(pkt)
                else:
                    while not self._stop.is_set():   # blocking put w/ stop check
                        try:
                            self.frame_buffer.put(pkt, timeout=0.1)
                            break
                        except queue.Full:
                            pass
                frame_id += 1
        finally:
            cap.release()
            self._push(self.frame_buffer, _SENTINEL)

    # --- Thread 2: detection ------------------------------------------------
    def _detection_loop(self):
        while not self._stop.is_set():
            pkt = self._pop(self.frame_buffer)
            if pkt is _SENTINEL:
                break
            if pkt is None:
                continue
            t0 = time.perf_counter()
            # imgsz downscales inside YOLO's letterbox; boxes come back
            # already scaled to the original frame size
            result = self.model.predict(pkt.frame, imgsz=self.imgsz,
                                        conf=self.conf, verbose=False,
                                        device=self.device)[0]
            infer_ms = (time.perf_counter() - t0) * 1000
            dets = sv.Detections.from_ultralytics(result)
            self._push(self.det_buffer,
                       DetPacket(pkt.frame_id, pkt.frame, dets, pkt.ts, infer_ms))
        self._push(self.det_buffer, _SENTINEL)

    # --- Thread 3: tracking --------------------------------------------------
    def _tracking_loop(self):
        names = self.model.names
        while not self._stop.is_set():
            pkt = self._pop(self.det_buffer)
            if pkt is _SENTINEL:
                break
            if pkt is None:
                continue
            tracked = self.tracker.update_with_detections(pkt.detections)
            tracks = [
                {"id": int(tid), "class": names[int(cls)],
                 "conf": float(cf), "bbox": xyxy.tolist()}
                for xyxy, cf, cls, tid in zip(tracked.xyxy, tracked.confidence,
                                              tracked.class_id, tracked.tracker_id)
            ]
            self._push(self.track_buffer,
                       TrackPacket(pkt.frame_id, pkt.frame, pkt.ts,
                                   pkt.infer_ms, tracks))
        self._push(self.track_buffer, _SENTINEL)

    # --- Thread 4 (optional): KITE intelligence -----------------------------
    def _intel_loop(self):
        from kinematics import TrackHistoryStore
        from classifier import fuse
        try:
            from threat_engine import ThreatEngine
            engine = ThreatEngine(self.zones)
        except ImportError:                # Phase 1: no threat engine yet
            engine = None

        self.engine = engine           # exposed for live zone updates (API)
        store = TrackHistoryStore()
        frame_size = None
        while not self._stop.is_set():
            pkt = self._pop(self.track_buffer)
            if pkt is _SENTINEL:
                break
            if pkt is None:
                continue
            t0 = time.perf_counter()
            if frame_size is None:
                h, w = pkt.frame.shape[:2]
                frame_size = (float(w), float(h))
            # media time for files, wall clock for live sources
            kin_ts = (pkt.frame_id / self.video_fps
                      if not self.live else pkt.ts)
            feats = store.update(pkt.tracks, kin_ts, frame_size)
            events = []
            for tr in pkt.tracks:
                f = feats.get(tr["id"])
                fused = fuse(tr["class"], tr["conf"], f)
                tr["fused_class"] = fused.label
                tr["fused_conf"] = fused.confidence
                tr["reason"] = fused.reason
                tr["flagged"] = fused.flagged
                tr["kinematics"] = f
                tr["trail"] = store.trail(tr["id"])
            if engine is not None:
                events = engine.update(pkt.tracks, kin_ts, frame_size)
            intel_ms = (time.perf_counter() - t0) * 1000
            self._push(self.intel_buffer,
                       IntelPacket(pkt.frame_id, pkt.frame, pkt.ts,
                                   pkt.infer_ms, intel_ms, pkt.tracks, events))
        self._push(self.intel_buffer, _SENTINEL)

    # --- queue helpers that respect the stop flag ----------------------------
    def _push(self, q, item):
        if self.live:
            q.put(item)
            return
        while not self._stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _pop(self, q):
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            return None

    # --- lifecycle ------------------------------------------------------------
    def start(self):
        loops = [self._camera_loop, self._detection_loop, self._tracking_loop]
        if self.intel:
            loops.append(self._intel_loop)
        for fn in loops:
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self.threads:
            t.join(timeout=2)

    def results(self):
        """Yield TrackPackets (or IntelPackets with intel=True) until the
        video ends (or forever for webcam)."""
        out_buffer = self.intel_buffer if self.intel else self.track_buffer
        while True:
            pkt = self._pop(out_buffer)
            if pkt is _SENTINEL:
                return
            if pkt is not None:
                yield pkt


# ---------------------------------------------------------------------------
# Sequential baseline: identical work, one thread (for the FPS comparison)
# ---------------------------------------------------------------------------

def sequential_results(model_path, source, imgsz, conf, device=None):
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                  imgsz=imgsz, verbose=False, device=device)   # warmup
    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(source)
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = time.perf_counter()
        t0 = time.perf_counter()
        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False,
                               device=device)[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        tracked = tracker.update_with_detections(sv.Detections.from_ultralytics(result))
        tracks = [
            {"id": int(tid), "class": model.names[int(cls)],
             "conf": float(cf), "bbox": xyxy.tolist()}
            for xyxy, cf, cls, tid in zip(tracked.xyxy, tracked.confidence,
                                          tracked.class_id, tracked.tracker_id)
        ]
        yield TrackPacket(frame_id, frame, ts, infer_ms, tracks)
        frame_id += 1
    cap.release()


# ---------------------------------------------------------------------------
# Drawing + main
# ---------------------------------------------------------------------------

# BGR threat-level colors for KITE intel mode
THREAT_COLORS = {
    "NONE": (120, 120, 120), "LOW": (0, 200, 255), "MEDIUM": (0, 165, 255),
    "HIGH": (0, 80, 255), "CRITICAL": (0, 0, 255),
}


def draw(pkt, fps, zones=None, ticker=None):
    frame = pkt.frame.copy()
    intel = isinstance(pkt, IntelPacket)

    # keep-out zones (filled translucent + outline)
    if zones:
        overlay = frame.copy()
        for z in zones:
            poly = np.array(z["points"], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 0, 180))
            cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    for tr in pkt.tracks:
        x1, y1, x2, y2 = map(int, tr["bbox"])
        if intel:
            cls = tr.get("fused_class", tr["class"])
            conf = tr.get("fused_conf", tr["conf"])
            threat = tr.get("threat")
            color = (THREAT_COLORS[threat["level"]] if threat and cls == "drone"
                     else CLASS_COLORS.get(cls, (0, 255, 0)))
            # fading trajectory trail
            trail = tr.get("trail") or []
            for i in range(1, len(trail)):
                a = i / len(trail)
                c = tuple(int(ch * a) for ch in color)
                cv2.line(frame, tuple(map(int, trail[i - 1])),
                         tuple(map(int, trail[i])), c, 2)
            # Kalman predicted-path ghost (dotted)
            for px, py in (tr.get("predicted") or [])[::2]:
                cv2.circle(frame, (int(px), int(py)), 2, color, -1)
            label = f'#{tr["id"]} {cls} {conf:.2f}'
            if threat and cls == "drone" and threat["level"] != "NONE":
                label += f' [{threat["level"]} {threat["score"]}]'
            if tr.get("flagged"):
                label += " !"
        else:
            color = CLASS_COLORS.get(tr["class"], (0, 255, 0))
            label = f'#{tr["id"]} {tr["class"]} {tr["conf"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(frame, f"infer: {pkt.infer_ms:5.1f} ms", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if intel:
        cv2.putText(frame, f"intel: {pkt.intel_ms:5.2f} ms", (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # event ticker: last few events along the bottom edge
    if ticker:
        h = frame.shape[0]
        for i, ev in enumerate(list(ticker)[-4:]):
            txt = f'{ev["type"]} #{ev["track_id"]}'
            cv2.putText(frame, txt, (15, h - 15 - 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        THREAT_COLORS.get(ev.get("severity", "LOW"),
                                          (0, 200, 255)), 2)
    return frame


def _reencode_h264(src, dst):
    """Re-encode OpenCV's MPEG-4 output to H.264 so Windows' built-in
    player can play it. Uses the ffmpeg bundled with imageio-ffmpeg."""
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        os.replace(src, dst)
        print("note: install imageio-ffmpeg for a Windows-playable H.264 "
              "file; current file needs VLC to play.")
        return
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    dst], check=True)
    os.remove(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(ROOT, "test_assets", "drone_test.mp4"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "best_int8.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default=None,
                    help='inference device, e.g. "intel:gpu" for the iGPU')
    ap.add_argument("--sequential", action="store_true",
                    help="run single-threaded baseline instead of pipeline")
    ap.add_argument("--no-show", action="store_true",
                    help="no window; just measure FPS over the whole video")
    ap.add_argument("--save", default=None, metavar="OUT.mp4",
                    help="also write the annotated video (boxes + FPS "
                         "counter) to this file — screen recording not needed")
    ap.add_argument("--intel", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="enable the KITE intelligence stage (kinematics + "
                         "fused classification + threat/events)")
    ap.add_argument("--zones", default=None, metavar="ZONES.json",
                    help='keep-out polygons JSON: [{"name": "pad", '
                         '"points": [[x,y], ...]}, ...] (implies --intel)')
    args = ap.parse_args()

    zones = []
    if args.zones:
        import json
        with open(args.zones) as fh:
            zones = json.load(fh)
        args.intel = True

    source = int(args.source) if str(args.source).isdigit() else args.source
    mode = "SEQUENTIAL" if args.sequential else "THREADED (3-thread pipeline)"
    print(f"mode={mode}  model={os.path.basename(str(args.model))}  "
          f"imgsz={args.imgsz}  source={source}")

    if args.sequential:
        gen = sequential_results(args.model, source, args.imgsz, args.conf,
                                 device=args.device)
        pipeline = None
    else:
        pipeline = ThreadedPipeline(args.model, source, args.imgsz, args.conf,
                                    device=args.device, intel=args.intel,
                                    zones=zones)
        pipeline.start()
        gen = pipeline.results()

    writer = None
    ticker = []                        # rolling on-screen event log
    n, t_start = 0, time.perf_counter()
    fps_smooth, last = 0.0, time.perf_counter()
    try:
        for pkt in gen:
            n += 1
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            # exponential smoothing so the on-screen FPS is readable
            fps_smooth = inst if n == 1 else 0.9 * fps_smooth + 0.1 * inst
            if isinstance(pkt, IntelPacket) and pkt.events:
                ticker.extend(pkt.events)
                for ev in pkt.events:
                    print(f'  EVENT {ev["type"]} #{ev["track_id"]} '
                          f'[{ev["severity"]}]')
                ticker = ticker[-8:]
            if not args.no_show or args.save:
                annotated = draw(pkt, fps_smooth, zones=zones, ticker=ticker)
                if args.save:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save + ".tmp.mp4",
                            cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                    writer.write(annotated)
                if not args.no_show:
                    cv2.imshow(mode, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - t_start
        if pipeline:
            pipeline.stop()
        if writer is not None:
            writer.release()
            _reencode_h264(args.save + ".tmp.mp4", args.save)
            print(f"annotated video saved -> {args.save}")
        cv2.destroyAllWindows()

    if n:
        print(f"\nprocessed {n} frames in {elapsed:.1f}s  "
              f"->  AVG FPS: {n / elapsed:.2f}")
    return n / elapsed if n else 0.0


if __name__ == "__main__":
    main()
