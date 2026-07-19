# KITE Kinematics — Measured Results

Model: `best_int8.onnx` @ imgsz 640, conf 0.3.
Track-level calls (majority vote over a track's frames, tracks >= 10 frames).
Ground truth by clip content: `bird_*.mp4` contain birds, `drone_test.mp4` contains drones.

## Overhead (drone_test.mp4, full clip, per frame)

| Pipeline | FPS | intel p50 (ms) | intel p95 (ms) |
|---|---|---|---|
| --no-intel | 35.07 | — | — |
| --intel | 30.34 | 3.108 | 5.418 |

## Decision-threshold sweep (one pipeline pass, evaluated offline)

Fused P(drone) is recorded per frame per track; the drone-vs-bird call is a
majority vote of frames above the threshold. The appearance baseline is
degenerate on this data — it calls every track "drone" — so 100% drone
recall is trivial and balanced accuracy is the honest metric. Operating
point = max balanced accuracy with drone recall >= 80%.

| Threshold | Missed drones (of 21) | Bird false alarms (of 241) | Balanced acc |
|---|---|---|---|
| 0.30 | 1 | 237 | 0.484 |
| 0.35 | 2 | 223 | 0.490 |
| 0.40 | 3 | 197 | 0.520 |
| 0.45 | 3 | 170 | 0.576 |
| 0.50 | 4 | 135 | 0.625 |
| 0.55 | 4 | 106 | 0.685 **<- operating point** |
| 0.60 | 5 | 69 | 0.738 |
| 0.65 | 6 | 41 | 0.772 |
| 0.70 | 12 | 17 | 0.679 |
| 0.75 | 15 | 9 | 0.624 |

## Drone clip (recall — must not lose drones)

| Config | Tracks | Missed drones | Miss rate |
|---|---|---|---|
| Appearance-only | 21 | 0 | 0.0% |
| Fused (KITE) @ 0.55 | 21 | 4 | 19.0% |

## Bird clips (false alarms — birds called drone, threshold 0.55)

| Clip | Tracks | Appearance FA | Fused FA |
|---|---|---|---|
| bird_01.mp4 | 31 | 31 | 7 |
| bird_02.mp4 | 0 | 0 | 0 |
| bird_03.mp4 | 33 | 33 | 23 |
| bird_04.mp4 | 177 | 177 | 76 |
| **TOTAL** | **241** | **241 (100.0%)** | **106 (44.0%)** |

## Summary

- Intel stage overhead: p50 3.108 ms / p95 5.418 ms per frame; FPS 35.07 -> 30.34.
- False alarms (bird tracks called drone): 241/241 appearance-only -> 106/241 fused.
- Missed drones: 0/21 appearance-only -> 4/21 fused.

*Caveats: bird clips are unlabeled beyond clip-level content; camera is assumed
mostly static; this is a small validation set, not the WOSDETC benchmark.*
