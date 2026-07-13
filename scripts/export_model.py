"""
Export best.pt -> ONNX (FP32) -> ONNX (INT8 quantized).

Pipeline:
  1. Ultralytics exports the PyTorch model to ONNX (models/best.onnx).
  2. onnxruntime quantizes it to INT8 (models/best_int8.onnx), calibrated
     on frames extracted from the test video so the int8 scaling factors
     match the kind of images we'll actually run on.

Usage:
  python scripts/export_model.py                  # full export + quantize
  python scripts/export_model.py --skip-export    # only re-quantize
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT_MODEL = os.path.join(ROOT, "best.pt")
MODELS_DIR = os.path.join(ROOT, "models")
ONNX_FP32 = os.path.join(MODELS_DIR, "best.onnx")
ONNX_INT8 = os.path.join(MODELS_DIR, "best_int8.onnx")
VIDEO = os.path.join(ROOT, "test_assets", "drone_test.mp4")
CALIB_DIR = os.path.join(ROOT, "scripts", "calibration_images")

IMGSZ = 640          # model was trained at 640; export matches it
CALIB_FRAMES = 100   # ~100 images is the standard amount for calibration


def export_fp32():
    """Step 1: PyTorch -> ONNX using ultralytics' built-in exporter."""
    from ultralytics import YOLO

    model = YOLO(PT_MODEL)
    # dynamic=True keeps input dims flexible so we can run imgsz=416 later
    # without re-exporting. opset 13+ is required for per-channel INT8.
    path = model.export(format="onnx", imgsz=IMGSZ, dynamic=True,
                        simplify=True, opset=13)
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.abspath(path) != os.path.abspath(ONNX_FP32):
        os.replace(path, ONNX_FP32)
    print(f"[export] FP32 ONNX -> {ONNX_FP32}")


def export_openvino():
    """Export to OpenVINO IR (Intel's inference engine — the Intel
    equivalent of TensorRT). half=True stores FP16 weights, which the
    Iris Xe iGPU runs natively; CPU silently upcasts where needed."""
    from ultralytics import YOLO

    model = YOLO(PT_MODEL)
    path = model.export(format="openvino", imgsz=IMGSZ, dynamic=True, half=True)
    dest = os.path.join(MODELS_DIR, "best_openvino_model")
    if os.path.abspath(path) != os.path.abspath(dest):
        if os.path.isdir(dest):
            import shutil
            shutil.rmtree(dest)
        os.replace(path, dest)
    print(f"[export] OpenVINO IR -> {dest}")


def extract_calibration_frames():
    """Pull evenly-spaced frames out of the test video for calibration."""
    os.makedirs(CALIB_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
    if len(existing) >= CALIB_FRAMES:
        print(f"[calib] reusing {len(existing)} existing frames")
        return existing

    if not os.path.exists(VIDEO):
        return existing  # fall back to whatever is there (maybe nothing)

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // CALIB_FRAMES)
    saved = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(CALIB_DIR, f"frame_{i:05d}.jpg"), frame)
        saved += 1
    cap.release()
    print(f"[calib] extracted {saved} frames from {os.path.basename(VIDEO)}")
    return glob.glob(os.path.join(CALIB_DIR, "*.jpg"))


def letterbox(img, size):
    """Resize keeping aspect ratio, pad with gray to a square `size` —
    the exact preprocessing YOLO uses, so calibration sees real inputs."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def quantize_int8(calib_images):
    """Step 2: FP32 ONNX -> INT8 ONNX via static quantization."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class VideoFrameReader(CalibrationDataReader):
        """Feeds preprocessed calibration frames to the quantizer,
        preprocessed identically to real inference (letterbox, RGB, /255)."""

        def __init__(self, paths):
            self.paths = list(paths)
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.paths):
                return None
            img = cv2.imread(self.paths[self.idx])
            self.idx += 1
            img = letterbox(img, IMGSZ)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = img.astype(np.float32) / 255.0        # 0..1
            blob = blob.transpose(2, 0, 1)[None]          # HWC -> NCHW
            return {"images": blob}

    # Pre-process fixes shapes/ops so the quantizer can handle every node.
    # skip_symbolic_shape: symbolic inference chokes on dynamic axes
    # (we exported with dynamic=True); plain ONNX shape inference suffices.
    pre = ONNX_FP32.replace(".onnx", "_pre.onnx")
    quant_pre_process(ONNX_FP32, pre, skip_symbolic_shape=True)

    # The detection head (model.23: DFL + box decode) is numerically fragile —
    # quantizing it collapses all detections to zero. Exclude it and quantize
    # only the backbone/neck, which is where ~90% of the compute lives.
    import onnx
    head_nodes = [n.name for n in onnx.load(pre).graph.node
                  if "/model.23" in n.name]

    if calib_images:
        print(f"[quant] static INT8 with {len(calib_images)} calibration frames "
              f"(excluding {len(head_nodes)} detect-head nodes)...")
        quantize_static(
            pre, ONNX_INT8,
            calibration_data_reader=VideoFrameReader(calib_images),
            quant_format=QuantFormat.QDQ,      # QDQ = best supported on CPU
            per_channel=True,                  # per-channel scales = less accuracy loss
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            nodes_to_exclude=head_nodes,
        )
    else:
        print("[quant] WARNING: no calibration frames found — falling back to "
              "dynamic quantization (often slower on conv nets).")
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(pre, ONNX_INT8, weight_type=QuantType.QInt8)

    os.remove(pre)
    print(f"[quant] INT8 ONNX -> {ONNX_INT8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse existing FP32 ONNX, only quantize")
    ap.add_argument("--openvino-only", action="store_true",
                    help="only export the OpenVINO IR model")
    args = ap.parse_args()

    if args.openvino_only:
        export_openvino()
        return

    if not args.skip_export:
        export_fp32()
    elif not os.path.exists(ONNX_FP32):
        sys.exit("no FP32 ONNX found — run without --skip-export first")

    calib = extract_calibration_frames()
    quantize_int8(calib)
    export_openvino()

    print("\n=== Model sizes ===")
    for p in (PT_MODEL, ONNX_FP32, ONNX_INT8):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):18s} {os.path.getsize(p) / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
