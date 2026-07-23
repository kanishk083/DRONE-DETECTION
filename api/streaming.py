"""
KITE Phase 3 — live video-intelligence streaming.

POST /stream/start            upload a video -> {session_id, ...}
WS   /ws/stream/{session_id}  one IntelPacket JSON per processed frame
POST /stream/{session_id}/zones  update keep-out polygons live
DELETE /stream/{session_id}   stop and clean up

Runs the same ThreadedPipeline + KITE intel stage the CLI demo uses.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import os
import sys
import tempfile
import time
import uuid

import cv2
from fastapi import (APIRouter, File, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)

from .schemas import StartStreamResponse, ZonesUpdate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODEL_PATH = os.getenv("STREAM_MODEL_PATH",
                       os.path.join(ROOT, "models", "best_int8.onnx"))
IMGSZ = int(os.getenv("STREAM_IMGSZ", "416"))
CONF = float(os.getenv("STREAM_CONF", "0.3"))
JPEG_QUALITY = 70
MAX_STREAM_WIDTH = 960          # downscale frames for the wire, not for YOLO
MAX_UPLOAD_MB = 200
MAX_SESSIONS = 3

router = APIRouter()
sessions: dict[str, "StreamSession"] = {}


class StreamSession:
    def __init__(self, video_path: str):
        from pipeline_demo import ThreadedPipeline  # deferred: heavy import

        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("could not open uploaded video")
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.pipeline = ThreadedPipeline(MODEL_PATH, video_path, imgsz=IMGSZ,
                                         conf=CONF, live=False, intel=True)
        self.frames_processed = 0
        self.running = False

    def start(self):
        self.pipeline.start()
        self.running = True

    def set_zones(self, zones: list[dict]):
        # engine is created inside the intel thread; may not exist yet
        engine = getattr(self.pipeline, "engine", None)
        if engine is not None:
            engine.set_zones(zones)
            return True
        self.pipeline.zones = zones      # picked up when the thread starts
        return False

    def stop(self):
        self.running = False
        self.pipeline.stop()
        try:
            os.remove(self.video_path)
        except OSError:
            pass


def _encode_frame(frame) -> str:
    h, w = frame.shape[:2]
    if w > MAX_STREAM_WIDTH:
        scale = MAX_STREAM_WIDTH / w
        frame = cv2.resize(frame, (MAX_STREAM_WIDTH, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def _packet_json(pkt, fps: float, orig_size) -> dict:
    tracks = []
    for tr in pkt.tracks:
        k = tr.get("kinematics")
        tracks.append({
            "id": tr["id"],
            "class": tr.get("fused_class", tr["class"]),
            "appearance_class": tr["class"],
            "appearance_conf": round(tr["conf"], 3),
            "fused_conf": round(tr.get("fused_conf", tr["conf"]), 3),
            "reason": tr.get("reason", ""),
            "flagged": tr.get("flagged", False),
            "bbox": [round(v, 1) for v in tr["bbox"]],
            "trail": [[round(x, 1), round(y, 1)] for x, y in tr.get("trail", [])],
            "predicted": [[round(x, 1), round(y, 1)]
                          for x, y in tr.get("predicted", [])],
            "kinematics": ({key: (round(val, 4) if isinstance(val, float) else val)
                            for key, val in dataclasses.asdict(k).items()}
                           if k is not None else None),
            "threat": tr.get("threat"),
        })
    return {
        "frame_id": pkt.frame_id,
        "ts": pkt.ts,
        "fps": round(fps, 1),
        "infer_ms": round(pkt.infer_ms, 1),
        "intel_ms": round(pkt.intel_ms, 2),
        "image_size": list(orig_size),
        "frame_jpeg_b64": _encode_frame(pkt.frame),
        "tracks": tracks,
        "events": pkt.events,
    }


@router.post("/stream/start", response_model=StartStreamResponse)
async def start_stream(file: UploadFile = File(...)):
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(429, "too many active sessions")
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(400, "invalid file type, must be a video")

    fd, path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or "")[1]
                                or ".mp4")
    size = 0
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                out.close()
                os.remove(path)
                raise HTTPException(413, f"video larger than {MAX_UPLOAD_MB} MB")
            out.write(chunk)

    try:
        # pipeline construction loads the model — off the event loop
        session = await asyncio.to_thread(StreamSession, path)
    except ValueError as e:
        os.remove(path)
        raise HTTPException(400, str(e))

    sid = uuid.uuid4().hex[:12]
    sessions[sid] = session
    return StartStreamResponse(session_id=sid, video_fps=session.video_fps,
                               frame_count=session.frame_count,
                               width=session.width, height=session.height)


@router.post("/stream/{sid}/zones")
async def update_zones(sid: str, body: ZonesUpdate):
    session = sessions.get(sid)
    if session is None:
        raise HTTPException(404, "unknown session")
    zones = [{"name": z.name, "points": [list(p) for p in z.points]}
             for z in body.zones]
    live = session.set_zones(zones)
    return {"ok": True, "live": live, "zones": len(zones)}


@router.delete("/stream/{sid}")
async def stop_stream(sid: str):
    session = sessions.pop(sid, None)
    if session is None:
        raise HTTPException(404, "unknown session")
    await asyncio.to_thread(session.stop)
    return {"ok": True}


@router.websocket("/ws/stream/{sid}")
async def stream_ws(ws: WebSocket, sid: str):
    session = sessions.get(sid)
    if session is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    session.start()
    gen = session.pipeline.results()
    orig_size = (session.width, session.height)
    frame_interval = 1.0 / session.video_fps
    fps_smooth, last = 0.0, time.perf_counter()
    try:
        while True:
            pkt = await asyncio.to_thread(next, gen, None)
            if pkt is None:               # video finished
                await ws.send_json({"done": True,
                                    "frames": session.frames_processed})
                break
            session.frames_processed += 1
            # pace playback to the video's native fps when we outrun it
            gap = time.perf_counter() - last
            if gap < frame_interval:
                await asyncio.sleep(frame_interval - gap)
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)   # true send-to-send fps
            fps_smooth = inst if fps_smooth == 0 else \
                0.9 * fps_smooth + 0.1 * inst
            last = now
            await ws.send_json(_packet_json(pkt, fps_smooth, orig_size))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sessions.pop(sid, None)
        await asyncio.to_thread(session.stop)
