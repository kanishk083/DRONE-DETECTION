"""
The headline benchmark: FPS on the test video, one optimization at a time.

Each stage stacks a technique on top of the previous one, so the table shows
exactly how much each idea contributes:

  BASELINE        best.pt        640   sequential
  +ONNX Runtime   best.onnx      640   sequential
  +INT8 quant     best_int8.onnx 640   sequential
  +Downscale      best_int8.onnx 416   sequential
  +3-thread pipe  best_int8.onnx 416   threaded

Usage:
  python scripts/benchmark_video.py
  python scripts/benchmark_video.py --max-frames 200     # quicker run
"""

import argparse
import os
import platform
import statistics
import time

import cv2

from pipeline_demo import ThreadedPipeline, sequential_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, "test_assets", "drone_test.mp4")
PT = os.path.join(ROOT, "best.pt")
ONNX = os.path.join(ROOT, "models", "best.onnx")
INT8 = os.path.join(ROOT, "models", "best_int8.onnx")
REPORT = os.path.join(ROOT, "OPTIMIZATION_RESULTS.md")

OV = os.path.join(ROOT, "models", "best_openvino_model")

STAGES = [
    # (label,                    model, imgsz, threaded, device)
    ("BASELINE (PyTorch)",        PT,   640,   False,    None),
    ("+ ONNX Runtime",            ONNX, 640,   False,    None),
    ("+ INT8 quantized",          INT8, 640,   False,    None),
    ("+ Downscale 416",           INT8, 416,   False,    None),
    ("+ 3-thread pipeline",       INT8, 416,   True,     None),
]

# OpenVINO stages (Intel's TensorRT equivalent) — added if the IR exists.
if os.path.isdir(OV):
    STAGES += [
        ("OpenVINO CPU",              OV, 640, False, None),
        ("OpenVINO iGPU",             OV, 640, False, "intel:gpu"),
        ("OpenVINO CPU + threads",    OV, 640, True,  None),
        ("OpenVINO iGPU + threads",   OV, 640, True,  "intel:gpu"),
        ("OpenVINO CPU 416 + threads", OV, 416, True,  None),
    ]


def run_stage(model, imgsz, threaded, max_frames, conf=0.3, device=None):
    """Process the video, return (avg_fps, per-frame inference latencies)."""
    latencies = []
    n = 0
    if threaded:
        pipe = ThreadedPipeline(model, VIDEO, imgsz=imgsz, conf=conf,
                                live=False, device=device)
        pipe.start()
        gen = pipe.results()
    else:
        pipe = None
        gen = sequential_results(model, VIDEO, imgsz, conf, device=device)

    t0 = time.perf_counter()
    for pkt in gen:
        latencies.append(pkt.infer_ms)
        n += 1
        if n >= max_frames:
            break
    elapsed = time.perf_counter() - t0
    if pipe:
        pipe.stop()
    return (n / elapsed if elapsed else 0.0), latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=300,
                    help="frames per stage (whole video can take a while on CPU)")
    args = ap.parse_args()

    assert os.path.exists(VIDEO), f"test video missing: {VIDEO}"
    for p in (PT, ONNX, INT8):
        assert os.path.exists(p), f"model missing: {p} (run export_model.py)"

    cap = cv2.VideoCapture(VIDEO)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print(f"video: {w}x{h} @ {src_fps:.0f}fps, {total} frames "
          f"(benchmarking first {min(args.max_frames, total)})\n")

    rows = []
    baseline_fps = None
    for label, model, imgsz, threaded, device in STAGES:
        print(f"running: {label:26s} ...", end=" ", flush=True)
        try:
            fps, lat = run_stage(model, imgsz, threaded, args.max_frames,
                                 device=device)
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {str(e)[:80]})")
            continue
        if baseline_fps is None:
            baseline_fps = fps
        p50 = statistics.median(lat)
        p95 = statistics.quantiles(lat, n=20)[-1] if len(lat) >= 20 else max(lat)
        speedup = fps / baseline_fps
        rows.append((label, os.path.basename(model), imgsz,
                     "3-thread" if threaded else "single", fps, p50, p95, speedup))
        print(f"{fps:6.2f} FPS  (x{speedup:.2f})")

    # ---- markdown table ----
    lines = [
        "| Stage | Model | imgsz | Threads | FPS | infer p50 (ms) | infer p95 (ms) | Speedup |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | "
                     f"**{r[4]:.2f}** | {r[5]:.1f} | {r[6]:.1f} | x{r[7]:.2f} |")
    table = "\n".join(lines)

    print("\n" + table)

    stamp = time.strftime("%Y-%m-%d %H:%M")
    block = (f"\n\n## Benchmark run — {stamp}\n\n"
             f"- Machine: {platform.processor()}\n"
             f"- Video: {w}x{h} @ {src_fps:.0f}fps, first "
             f"{min(args.max_frames, total)} frames, conf=0.3\n\n{table}\n")
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"\nappended results to {REPORT}")


if __name__ == "__main__":
    main()
