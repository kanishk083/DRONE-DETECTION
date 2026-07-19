"""
Appearance-model evaluation on the Roboflow drone-vs-bird COCO test split.

Independent still images — no motion available — so this measures exactly
what KITE's appearance-only baseline can do. For every ground-truth box,
the best IoU-matched detection (IoU >= 0.3) supplies the predicted class.

Usage:
  python scripts/eval_dataset.py                      # test split
  python scripts/eval_dataset.py --split valid
"""

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "drone-vs-bird")

# COCO category_id -> semantic class (verified visually: cat 1 = bird,
# cat 2 = drone; see KINEMATICS_RESULTS.md dataset note)
CAT_TO_CLASS = {1: "bird", 2: "drone"}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "best_int8.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model, task="detect")

    split_dir = os.path.join(DATA, args.split)
    coco = json.load(open(os.path.join(split_dir, "_annotations.coco.json")))
    imgs = {i["id"]: i for i in coco["images"]}
    gt_by_img = {}
    for a in coco["annotations"]:
        if a["category_id"] not in CAT_TO_CLASS:
            continue
        x, y, w, h = a["bbox"]
        gt_by_img.setdefault(a["image_id"], []).append(
            (CAT_TO_CLASS[a["category_id"]], [x, y, x + w, y + h]))

    confusion = Counter()          # (gt, pred) including (gt, "missed")
    n_done = 0
    for img_id, gts in gt_by_img.items():
        path = os.path.join(split_dir, imgs[img_id]["file_name"])
        result = model.predict(path, imgsz=args.imgsz, conf=args.conf,
                               verbose=False)[0]
        dets = [(model.names[int(b.cls[0])], b.xyxy[0].tolist())
                for b in result.boxes]
        for gt_cls, gt_box in gts:
            best, best_iou = None, 0.3
            for d_cls, d_box in dets:
                v = iou(gt_box, d_box)
                if v >= best_iou:
                    best, best_iou = d_cls, v
            confusion[(gt_cls, best or "missed")] += 1
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(gt_by_img)} images ...")

    print(f"\nsplit={args.split}  model={os.path.basename(args.model)} "
          f"@{args.imgsz} conf={args.conf}")
    print(f"{'GT class':<10}{'as drone':>10}{'as bird':>10}{'missed':>10}"
          f"{'class acc':>12}")
    for cls in ("drone", "bird"):
        as_d = confusion[(cls, "drone")]
        as_b = confusion[(cls, "bird")]
        miss = confusion[(cls, "missed")]
        total = as_d + as_b + miss
        correct = as_d if cls == "drone" else as_b
        acc = correct / total if total else 0.0
        print(f"{cls:<10}{as_d:>10}{as_b:>10}{miss:>10}{acc:>11.1%}")


if __name__ == "__main__":
    main()
