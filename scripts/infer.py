"""
infer.py — Inference for Oil Storage Tank Detection
=====================================================
Handles both normal-sized tiles (640×640) and arbitrarily large
satellite images via automatic tiling + NMS merge.

BONUS: Optional circle-fitting post-processing.
  Oil storage tanks are near-perfect circles from overhead view.
  After bounding box detection, we fit a Hough circle to each
  detected region to refine the shape estimate and filter
  non-circular false positives.

Usage:
    # Single image (640×640 tile)
    python scripts/infer.py --weights best.pt --source tile.jpg

    # Large satellite image (auto-tiled)
    python scripts/infer.py --weights best.pt --source full_scene.jpg

    # With circle post-processing
    python scripts/infer.py --weights best.pt --source scene.jpg --circle-fit

    # Batch inference on a folder
    python scripts/infer.py --weights best.pt --source data/processed/images/test/
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TILE_SIZE    = 640
TILE_OVERLAP = 64
NMS_IOU_THR  = 0.45

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#00d4ff"
RED      = "#ff4d6d"
GREEN    = "#00e676"
TEXT_COL = "#e0e0e0"
GRID_COL = "#21262d"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def box_iou(a: list[float], b: list[float]) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1-ix0)*(iy1-iy0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-9)


def nms(boxes: list[dict], iou_thr: float = NMS_IOU_THR) -> list[dict]:
    """Greedy NMS. Each box: {box:[x1,y1,x2,y2], conf:float}"""
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    kept = []
    suppressed = set()
    for i, b in enumerate(boxes_sorted):
        if i in suppressed:
            continue
        kept.append(b)
        for j in range(i+1, len(boxes_sorted)):
            if j not in suppressed and box_iou(b["box"], boxes_sorted[j]["box"]) > iou_thr:
                suppressed.add(j)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Circle fitting post-processing
# ─────────────────────────────────────────────────────────────────────────────
def fit_circle_in_region(
    img_gray: np.ndarray,
    box: list[float],
    circularity_thresh: float = 0.45,
) -> dict | None:
    """
    Attempt to fit a Hough circle within a detected bounding box region.

    Oil storage tanks are near-perfect circles from overhead.
    This step:
      1. Crops the detected region
      2. Applies Canny edge detection
      3. Runs HoughCircles
      4. If a strong circle is found, returns its parameters
      5. If no circle found (low circularity), marks as potential FP

    Returns:
        dict with {cx, cy, radius, circularity_score} or None if no circle
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = 8
    x1c = max(0, x1-pad); y1c = max(0, y1-pad)
    x2c = min(img_gray.shape[1], x2+pad); y2c = min(img_gray.shape[0], y2+pad)
    crop = img_gray[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None

    # Preprocess: blur + CLAHE for edge clarity
    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(blurred)

    # Estimate expected radius range from bbox size
    w_box  = x2 - x1
    h_box  = y2 - y1
    r_min  = max(5, int(min(w_box, h_box) * 0.3))
    r_max  = int(max(w_box, h_box) * 0.7)

    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(5, int(min(w_box, h_box) * 0.5)),
        param1=60,   # Canny high threshold
        param2=18,   # accumulator threshold (lower = more permissive)
        minRadius=r_min,
        maxRadius=r_max,
    )

    if circles is None:
        # No circle found — check circularity from contour as fallback
        edges    = cv2.Canny(enhanced, 30, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = 4 * math.pi * area / (peri**2 + 1e-6)
        if circ < circularity_thresh:
            return None
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        return {
            "cx": int(cx) + x1c, "cy": int(cy) + y1c,
            "radius": int(r), "circularity_score": round(circ, 3),
            "source": "contour",
        }

    # Pick the most central circle
    cx_crop, cy_crop, r = circles[0][0]
    circ_score = min(1.0, r**2 * math.pi / ((w_box * h_box) + 1e-6))
    return {
        "cx": int(cx_crop) + x1c, "cy": int(cy_crop) + y1c,
        "radius": int(r), "circularity_score": round(float(circ_score), 3),
        "source": "hough",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tiled inference for large satellite images
# ─────────────────────────────────────────────────────────────────────────────
def tile_predict(
    model, img_bgr: np.ndarray, conf: float = 0.25
) -> list[dict]:
    """
    Tile a large image, run detection per tile, return merged detections.
    """
    h, w = img_bgr.shape[:2]
    stride = TILE_SIZE - TILE_OVERLAP

    xs = list(range(0, max(1, w - TILE_SIZE + 1), stride))
    ys = list(range(0, max(1, h - TILE_SIZE + 1), stride))
    if not xs or xs[-1] + TILE_SIZE < w: xs.append(max(0, w - TILE_SIZE))
    if not ys or ys[-1] + TILE_SIZE < h: ys.append(max(0, h - TILE_SIZE))
    xs = sorted(set(xs)); ys = sorted(set(ys))

    raw_boxes = []
    for y0 in ys:
        for x0 in xs:
            tile = img_bgr[y0: y0+TILE_SIZE, x0: x0+TILE_SIZE]
            res  = model.predict(tile, conf=conf, verbose=False)
            if res and res[0].boxes is not None:
                for box, cs in zip(res[0].boxes.xyxy.cpu().numpy(),
                                   res[0].boxes.conf.cpu().numpy()):
                    raw_boxes.append({
                        "box": [box[0]+x0, box[1]+y0, box[2]+x0, box[3]+y0],
                        "conf": float(cs),
                    })

    return nms(raw_boxes)


def predict_image(
    model, img_path: Path, conf: float = 0.25
) -> list[dict]:
    """Auto-detect tile vs. large image and run appropriate inference."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    if h > TILE_SIZE or w > TILE_SIZE:
        return tile_predict(model, img, conf)

    res = model.predict(str(img_path), conf=conf, verbose=False)
    if not res or res[0].boxes is None:
        return []
    return [
        {"box": box.tolist(), "conf": float(cs)}
        for box, cs in zip(res[0].boxes.xyxy.cpu().numpy(),
                           res[0].boxes.conf.cpu().numpy())
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def visualize(
    img_path: Path,
    detections: list[dict],
    circles: list[dict | None],
    out_path: Path,
    pixel_res_m: float = 1.2,
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.imshow(img_rgb)

    for det, circ in zip(detections, circles):
        b = det["box"]; c = det["conf"]
        col = GREEN if c >= 0.75 else (ACCENT if c >= 0.50 else RED)

        # Bounding box
        rect = plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                              lw=1.8, edgecolor=col, facecolor="none")
        ax.add_patch(rect)
        ax.text(b[0], b[1]-6, f"{c:.2f}", color=col, fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=DARK_BG, alpha=0.8, edgecolor="none"))

        # Circle overlay (if fitted)
        if circ is not None:
            r_m = circ["radius"] * pixel_res_m
            circle_patch = plt.Circle(
                (circ["cx"], circ["cy"]), circ["radius"],
                color="#ffd700", fill=False, lw=1.5, linestyle="--"
            )
            ax.add_patch(circle_patch)
            ax.text(circ["cx"], circ["cy"] + circ["radius"] + 8,
                    f"r≈{r_m:.0f}m", color="#ffd700", fontsize=6.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=DARK_BG, alpha=0.75, edgecolor="none"))

    n = len(detections)
    n_circ = sum(1 for c in circles if c is not None)
    ax.set_title(
        f"{img_path.name} — {n} tank{'s' if n!=1 else ''} detected"
        + (f"  |  {n_circ} circle fits" if any(circles) else ""),
        color=TEXT_COL, fontsize=10, pad=6
    )

    handles = [
        mpatches.Patch(color=GREEN,    label="High conf (≥0.75)"),
        mpatches.Patch(color=ACCENT,   label="Mid conf (0.50-0.75)"),
        mpatches.Patch(color=RED,      label="Low conf (<0.50)"),
    ]
    if any(circles):
        handles.append(mpatches.Patch(color="#ffd700", label="Circle fit (Hough)", fill=False))
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              facecolor=DARK_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    weights:     Path,
    source:      Path,
    conf:        float = 0.25,
    output_dir:  Path  = Path("results/inference"),
    circle_fit:  bool  = False,
    pixel_res_m: float = 1.2,
) -> list[dict]:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model     = YOLO(str(weights))
    img_paths = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png")) \
        if source.is_dir() else [source]

    if not img_paths:
        print(f"[WARN] No images found at {source}")
        return []

    print(f"[INFO] Model:  {weights.name}")
    print(f"[INFO] Source: {source} ({len(img_paths)} image(s))")
    print(f"[INFO] Circle fit: {'enabled' if circle_fit else 'disabled'}\n")

    all_results = []
    for img_path in tqdm(img_paths, unit="img"):
        detections = predict_image(model, img_path, conf)

        # Circle fitting (BONUS)
        circle_info = []
        if circle_fit:
            img_bgr  = cv2.imread(str(img_path))
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr is not None else None
            for det in detections:
                circ = fit_circle_in_region(img_gray, det["box"]) if img_gray is not None else None
                circle_info.append(circ)
        else:
            circle_info = [None] * len(detections)

        # Visualize
        out_vis = output_dir / f"{img_path.stem}_detected.jpg"
        visualize(img_path, detections, circle_info, out_vis, pixel_res_m)

        result = {
            "image": img_path.name,
            "num_detections": len(detections),
            "detections": [
                {
                    "box":  d["box"],
                    "conf": round(d["conf"], 4),
                    "circle": c,
                }
                for d, c in zip(detections, circle_info)
            ]
        }
        all_results.append(result)

    # Save JSON
    json_path = output_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    total = sum(r["num_detections"] for r in all_results)
    print(f"\n[INFO] Detected {total} tanks across {len(img_paths)} image(s)")
    print(f"[INFO] Saved to {output_dir}/")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run oil tank detection inference on images or folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/infer.py --weights best.pt --source image.jpg
  python scripts/infer.py --weights best.pt --source data/processed/images/test/
  python scripts/infer.py --weights best.pt --source scene.jpg --circle-fit
        """
    )
    parser.add_argument("--weights",       required=True)
    parser.add_argument("--source",        required=True)
    parser.add_argument("--conf",          type=float, default=0.25)
    parser.add_argument("--output-dir",    default="results/inference")
    parser.add_argument("--circle-fit",    action="store_true",
                        help="Apply Hough circle fitting to each detection (BONUS)")
    parser.add_argument("--pixel-res-m",   type=float, default=1.2,
                        help="Pixel resolution in metres/px (default: 1.2 for SPOT)")
    args = parser.parse_args()

    run_inference(
        weights     = Path(args.weights),
        source      = Path(args.source),
        conf        = args.conf,
        output_dir  = Path(args.output_dir),
        circle_fit  = args.circle_fit,
        pixel_res_m = args.pixel_res_m,
    )
