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
                 device=None):
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.source = source
        self.imgsz = imgsz
        self.conf = conf
        self.device = device            # e.g. "intel:gpu" for OpenVINO on iGPU
        # Webcam => live mode (drop frames to stay realtime).
        # Video file => process every frame (blocking queues, no drops).
        self.live = live if live is not None else isinstance(source, int)

        make_q = LatestQueue if self.live else lambda: queue.Queue(maxsize=32)
        self.frame_buffer = make_q()       # camera  -> detection
        self.det_buffer = make_q()         # detect  -> tracking
        self.track_buffer = make_q()       # track   -> main thread

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
        for fn in (self._camera_loop, self._detection_loop, self._tracking_loop):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self.threads:
            t.join(timeout=2)

    def results(self):
        """Yield TrackPackets until the video ends (or forever for webcam)."""
        while True:
            pkt = self._pop(self.track_buffer)
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

def draw(pkt, fps):
    frame = pkt.frame.copy()
    for tr in pkt.tracks:
        x1, y1, x2, y2 = map(int, tr["bbox"])
        color = CLASS_COLORS.get(tr["class"], (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'#{tr["id"]} {tr["class"]} {tr["conf"]:.2f}'
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(frame, f"infer: {pkt.infer_ms:5.1f} ms", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
    args = ap.parse_args()

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
                                    device=args.device)
        pipeline.start()
        gen = pipeline.results()

    writer = None
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
            if not args.no_show or args.save:
                annotated = draw(pkt, fps_smooth)
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
