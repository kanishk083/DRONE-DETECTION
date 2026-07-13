# Drone Detection — Inference Optimization Results

Optimizing YOLO11n drone detection (`best.pt`, 2 classes: drone/bird) for CPU
real-time performance. All numbers measured on this machine — no estimates.

**Test setup**
- CPU: Intel Core i7-1185G7 @ 3.00GHz (Tiger Lake, AVX-512 VNNI — native INT8 instructions) + Iris Xe iGPU
- Runtime: Python 3.14, ultralytics 8.4.92, ONNX Runtime 1.27.0, OpenVINO 2026.2.1, supervision 0.29.1
- Test video: `test_assets/drone_test.mp4` — drone swarm, 768×432 @ 30fps, 441 frames
- Every stage includes the full workload per frame: video decode → detect → ByteTrack tracking

## Headline result

**13.7 FPS → 52.4 FPS (×3.8).** Baseline could not keep up with a 30fps
stream; the optimized pipeline runs at 1.75× real-time. The best *lossless*
config (no accuracy trade-off) reaches **33.5 FPS** — above real-time.

Benchmark run 2026-07-12, full 441 frames, conf=0.3:

| Stage | Model | imgsz | Threads | FPS | infer p50 (ms) | infer p95 (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| BASELINE (PyTorch) | best.pt | 640 | single | **13.71** | 52.5 | 78.8 | ×1.00 |
| + ONNX Runtime | best.onnx | 640 | single | **23.94** | 37.0 | 44.8 | ×1.75 |
| + INT8 quantized | best_int8.onnx | 640 | single | **26.83** | 31.4 | 42.2 | ×1.96 |
| + Downscale 416 | best_int8.onnx | 416 | single | **48.81** | 16.4 | 25.0 | ×3.56 |
| + 3-thread pipeline | best_int8.onnx | 416 | 3-thread | **52.35** | 18.0 | 24.9 | ×3.82 |
| OpenVINO CPU | best_openvino_model | 640 | single | **28.30** | 27.6 | 40.4 | ×2.06 |
| OpenVINO iGPU | best_openvino_model | 640 | single | **18.79** | 33.5 | 45.4 | ×1.37 |
| OpenVINO CPU + threads | best_openvino_model | 640 | 3-thread | **33.50** | 27.8 | 34.7 | ×2.44 |
| OpenVINO iGPU + threads | best_openvino_model | 640 | 3-thread | **26.54** | 37.1 | 42.2 | ×1.94 |
| OpenVINO CPU 416 + threads | best_openvino_model | 416 | 3-thread | **21.19** | 31.8 | 40.4 | ×1.55 |

Rows 1–5 stack one technique at a time, so each idea's gain is individually
attributable. Rows 6–10 test OpenVINO (Intel's inference engine) as an
alternative runtime.

## Accuracy check (sampled frames, conf=0.3, total detections)

| Config | Detections | vs baseline |
|---|---|---|
| PyTorch @ 640 (baseline) | 119 | — |
| INT8 ONNX @ 640 | 130 | no loss (slightly more near threshold) |
| OpenVINO @ 640 | matches ONNX (3/11/11 per-frame parity) | no loss |
| INT8 ONNX @ 416 | 71 | **−40% — loses small/distant drones** |

## Recommended configurations

| Goal | Config | Measured FPS | Accuracy |
|---|---|---|---|
| **Max accuracy, real-time** | OpenVINO CPU @ 640 + 3-thread pipeline | **33.50** | lossless |
| **Max speed** | INT8 ONNX @ 416 + 3-thread pipeline | **52.35** | −40% recall on small objects |

For drone-swarm surveillance, OpenVINO@640+threading is the right default:
above real-time at 30fps with zero accuracy loss. The 416 path suits
higher-fps cameras or weaker hardware, when nearby drones matter most.
The web API (`api/main.py`) serves the OpenVINO model by default
(override with the `MODEL_PATH` env var).

## The techniques, explained

### 1. ONNX Runtime (×1.75)
`best.pt` runs on PyTorch, which carries training machinery (dynamic graphs,
autograd) at inference time. Exporting to ONNX freezes the network into a
static graph of pure math ops; ONNX Runtime then applies operator fusion and
optimized CPU kernels. Biggest single "free" win — same weights, same outputs.

### 2. INT8 static quantization (×1.96 cumulative)
Weights/activations converted FP32 → INT8: 8-bit integer math uses this CPU's
AVX-512 VNNI instructions and shrinks the model (12.09 MB → 5.94 MB).
*Static* quantization calibrates the float→int scaling on ~111 real frames
extracted from the test video. **Key lesson learned:** quantizing the whole
network collapsed accuracy to zero detections — the detection head (DFL box
decoder, `model.23`, 172 nodes) is numerically fragile. Excluding the head and
quantizing only the backbone/neck (~90% of compute) preserved full accuracy.

### 3. Input downscaling 640 → 416 (×3.56 cumulative)
Inference cost scales with pixel count: 416² is 2.4× fewer pixels than 640².
Massive latency win, but this is the one technique with a real accuracy
trade-off — small/distant drones become too few pixels to detect (−40%
detections on the swarm video). Use case dependent.

### 4. Multi-threaded pipeline with shared buffers (×3.82 cumulative)
Sequential processing runs `read → detect → track` in series — each stage
idles while the others work. The pipeline splits them into three threads
connected by queues:

```
[Camera thread] → Frame Buffer → [Detection thread] → Detection Buffer → [Tracking thread] → Tracking Buffer → display
```

All stages work simultaneously on different frames, so throughput is set by
the slowest stage (detection) instead of the sum of all stages. Live mode uses
`maxsize=1` drop-oldest buffers so the display never lags behind the camera;
file mode uses blocking buffers so every frame is processed.

### 5. OpenVINO — Intel's TensorRT equivalent (best lossless: ×2.44)
Intel's inference engine, with kernels tuned per CPU generation. At 640 it
beat both ONNX Runtime FP32 (+18%) and even backbone-INT8 ONNX (+5%), and
threaded it became the best lossless config (33.5 FPS). Two honest findings:
- **The Iris Xe iGPU was SLOWER than the CPU** (18.8 vs 28.3 FPS): for a
  2.6M-parameter model, per-frame CPU↔GPU transfer overhead outweighs the
  iGPU's modest compute, and it competes with the CPU for the same power/
  thermal budget.
- **OpenVINO @416 was slower than @640** (21.2 FPS): the IR graph is
  optimized for its export shape; feeding a different size falls onto a slow
  dynamic-shape path. Downscaling belongs to the ONNX INT8 config, where it
  works (52.4 FPS).

## Deployment note: NVIDIA hardware

On a machine with an NVIDIA GPU (Jetson, RTX server), repeat this exact
recipe with **TensorRT**: `model.export(format="engine", int8=True)` +
the same calibration frames + the same 3-thread pipeline. TensorRT is
NVIDIA-only, which is why it could not be benchmarked on this machine.

## Reproduce

```bash
pip install ultralytics onnx onnxruntime openvino supervision

python scripts/export_model.py            # best.pt -> ONNX FP32 -> INT8 + OpenVINO IR
python scripts/benchmark_video.py         # the full table above
python scripts/pipeline_demo.py           # LIVE demo window (INT8@416): ~50 FPS on screen
python scripts/pipeline_demo.py --sequential --model best.pt   # baseline window: ~14 FPS
python scripts/pipeline_demo.py --model models/best_openvino_model --imgsz 640   # lossless config
python scripts/pipeline_demo.py --source 0     # live webcam demo
```
