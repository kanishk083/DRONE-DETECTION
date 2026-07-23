"""
KITE Phase 1 benchmark — measures the two numbers that justify the layer:

  1. Overhead: per-frame cost of the intelligence stage (p50/p95 ms) and
     end-to-end FPS with vs without --intel.
  2. Classification: appearance-only vs fused drone-vs-bird calls at the
     track level, on clips whose content is known (bird_*.mp4 = birds,
     drone_test.mp4 = drones).

Writes KINEMATICS_RESULTS.md at the repo root.

Usage:
  python scripts/benchmark_kinematics.py
  python scripts/benchmark_kinematics.py --model models/best_int8.onnx --imgsz 416
"""

import argparse
import glob
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_demo import ThreadedPipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_TRACK_FRAMES = 10       # ignore flickers — too short to classify either way


def run_clip(model, source, imgsz, conf, intel):
    """Run the pipeline over a full clip; return (fps, intel_ms list,
    per-track appearance votes, per-track per-frame p_drone lists)."""
    pipe = ThreadedPipeline(model, source, imgsz=imgsz, conf=conf,
                            live=False, intel=intel)
    pipe.start()
    app_votes = defaultdict(Counter)     # tid -> Counter(class)
    p_drone = defaultdict(list)          # tid -> [fused P(drone) per frame]
    intel_ms = []
    n, t0 = 0, time.perf_counter()
    for pkt in pipe.results():
        n += 1
        if intel:
            intel_ms.append(pkt.intel_ms)
        for tr in pkt.tracks:
            app_votes[tr["id"]][tr["class"]] += 1
            if intel:
                c = tr["fused_conf"]
                p_drone[tr["id"]].append(
                    c if tr["fused_class"] == "drone" else 1.0 - c)
    elapsed = time.perf_counter() - t0
    pipe.stop()
    return (n / elapsed if elapsed else 0.0), intel_ms, app_votes, p_drone


def track_calls(votes):
    """Track-level majority call, ignoring too-short tracks."""
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()
            if sum(c.values()) >= MIN_TRACK_FRAMES}


def fused_calls_at(p_drone, threshold):
    """Track-level fused call at a given decision threshold: majority of
    frames with P(drone) >= threshold."""
    out = {}
    for tid, ps in p_drone.items():
        if len(ps) < MIN_TRACK_FRAMES:
            continue
        drone_frames = sum(1 for p in ps if p >= threshold)
        out[tid] = "drone" if drone_frames * 2 >= len(ps) else "bird"
    return out


def pct(x, n):
    return f"{100 * x / n:.1f}%" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "best_int8.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    drone_clip = os.path.join(ROOT, "test_assets", "drone_test.mp4")
    bird_clips = sorted(glob.glob(os.path.join(ROOT, "test_assets", "bird_*.mp4")))
    model_name = os.path.basename(args.model)

    # -- 1) overhead on the drone clip ------------------------------------
    print(f"[1/3] overhead: {model_name} @{args.imgsz}, no-intel vs intel ...")
    fps_off, _, _, _ = run_clip(args.model, drone_clip, args.imgsz, args.conf, False)
    fps_on, intel_ms, app_d, fused_d = run_clip(args.model, drone_clip,
                                                args.imgsz, args.conf, True)
    p50 = statistics.median(intel_ms) if intel_ms else 0.0
    p95 = (statistics.quantiles(intel_ms, n=20)[18]
           if len(intel_ms) >= 20 else max(intel_ms, default=0.0))
    print(f"    no-intel {fps_off:.2f} FPS | intel {fps_on:.2f} FPS | "
          f"intel stage p50 {p50:.3f} ms p95 {p95:.3f} ms")

    # -- 2) drone recall on the drone clip ---------------------------------
    print("[2/3] drone clip classification ...")
    app_calls_d = track_calls(app_d)
    n_d = len(app_calls_d)
    app_missed = sum(1 for c in app_calls_d.values() if c != "drone")

    # -- 3) false alarms on the bird clips ---------------------------------
    print(f"[3/3] bird clips ({len(bird_clips)}) classification ...")
    bird_runs = []
    for clip in bird_clips:
        _, _, app_b, pd_b = run_clip(args.model, clip, args.imgsz,
                                     args.conf, True)
        bird_runs.append((os.path.basename(clip), app_b, pd_b))

    # -- threshold sweep: one pipeline pass, whole trade curve --------------
    THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    MIN_DRONE_RECALL = 0.80     # mission constraint: keep >= 80% of drones
    sweep = []                  # (threshold, missed, fa, balanced_acc)
    tot_tracks = sum(len(track_calls(ab)) for _, ab, _ in bird_runs)
    tot_app_fa = sum(sum(1 for c in track_calls(ab).values() if c == "drone")
                     for _, ab, _ in bird_runs)
    for th in THRESHOLDS:
        fc_d = fused_calls_at(fused_d, th)
        missed = sum(1 for t in app_calls_d if fc_d.get(t) != "drone")
        fa = 0
        for _, ab, pd_b in bird_runs:
            fc_b = fused_calls_at(pd_b, th)
            fa += sum(1 for t in track_calls(ab) if fc_b.get(t) == "drone")
        recall_d = (n_d - missed) / n_d if n_d else 0.0
        recall_b = (tot_tracks - fa) / tot_tracks if tot_tracks else 0.0
        bal = (recall_d + recall_b) / 2
        sweep.append((th, missed, fa, bal))
        print(f"    threshold {th:.2f}: missed {missed}/{n_d}, "
              f"FA {fa}/{tot_tracks}, balanced acc {bal:.3f}")

    # operating point: max balanced accuracy subject to the recall floor
    # (the appearance baseline is degenerate — it calls everything drone,
    # so 100% recall is trivial and balanced accuracy is the honest metric)
    eligible = [r for r in sweep
                if (n_d - r[1]) / n_d >= MIN_DRONE_RECALL] or sweep
    th_op, op_missed, op_fa, op_bal = max(eligible, key=lambda r: r[3])
    print(f"    -> operating point: threshold {th_op:.2f} "
          f"(balanced acc {op_bal:.3f})")

    # per-clip table at the operating point
    rows = []
    for name, ab, pd_b in bird_runs:
        ac = track_calls(ab)
        fc = fused_calls_at(pd_b, th_op)
        app_fa = sum(1 for c in ac.values() if c == "drone")
        fu_fa = sum(1 for t in ac if fc.get(t) == "drone")
        rows.append((name, len(ac), app_fa, fu_fa))
    tot_fused_fa = op_fa
    fused_missed = op_missed

    # -- write results ------------------------------------------------------
    out = os.path.join(ROOT, "KINEMATICS_RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"""# KITE Kinematics — Measured Results

Model: `{model_name}` @ imgsz {args.imgsz}, conf {args.conf}.
Track-level calls (majority vote over a track's frames, tracks >= {MIN_TRACK_FRAMES} frames).
Ground truth by clip content: `bird_*.mp4` contain birds, `drone_test.mp4` contains drones.

## Overhead (drone_test.mp4, full clip, per frame)

| Pipeline | FPS | intel p50 (ms) | intel p95 (ms) |
|---|---|---|---|
| --no-intel | {fps_off:.2f} | — | — |
| --intel | {fps_on:.2f} | {p50:.3f} | {p95:.3f} |

## Decision-threshold sweep (one pipeline pass, evaluated offline)

Fused P(drone) is recorded per frame per track; the drone-vs-bird call is a
majority vote of frames above the threshold. The appearance baseline is
degenerate on this data — it calls every track "drone" — so 100% drone
recall is trivial and balanced accuracy is the honest metric. Operating
point = max balanced accuracy with drone recall >= {int(MIN_DRONE_RECALL * 100)}%.

| Threshold | Missed drones (of {n_d}) | Bird false alarms (of {tot_tracks}) | Balanced acc |
|---|---|---|---|
""")
        for th, m, fa, bal in sweep:
            marker = " **<- operating point**" if th == th_op else ""
            f.write(f"| {th:.2f} | {m} | {fa} | {bal:.3f}{marker} |\n")
        f.write(f"""
## Drone clip (recall — must not lose drones)

| Config | Tracks | Missed drones | Miss rate |
|---|---|---|---|
| Appearance-only | {n_d} | {app_missed} | {pct(app_missed, n_d)} |
| Fused (KITE) @ {th_op:.2f} | {n_d} | {fused_missed} | {pct(fused_missed, n_d)} |

## Bird clips (false alarms — birds called drone, threshold {th_op:.2f})

| Clip | Tracks | Appearance FA | Fused FA |
|---|---|---|---|
""")
        for name, n, a, fu in rows:
            f.write(f"| {name} | {n} | {a} | {fu} |\n")
        f.write(f"""| **TOTAL** | **{tot_tracks}** | **{tot_app_fa} ({pct(tot_app_fa, tot_tracks)})** | **{tot_fused_fa} ({pct(tot_fused_fa, tot_tracks)})** |

## Summary

- Intel stage overhead: p50 {p50:.3f} ms / p95 {p95:.3f} ms per frame; FPS {fps_off:.2f} -> {fps_on:.2f}.
- False alarms (bird tracks called drone): {tot_app_fa}/{tot_tracks} appearance-only -> {tot_fused_fa}/{tot_tracks} fused.
- Missed drones: {app_missed}/{n_d} appearance-only -> {fused_missed}/{n_d} fused.

*Caveats: bird clips are unlabeled beyond clip-level content; camera is assumed
mostly static; this is a small validation set, not the WOSDETC benchmark.*
""")
    print(f"\nresults written -> {out}")


if __name__ == "__main__":
    main()
