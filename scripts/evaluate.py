"""
evaluate.py — Comprehensive Evaluation for Oil Storage Tank Detection
======================================================================
Evaluates on the TEST SET ONLY (never training or validation).

Produces:
  ┌─────────────────────────────────────────────────────────┐
  │  Metric               Source                            │
  │  ─────────────────── ─────────────────────────────────  │
  │  Precision            Ultralytics val() + manual IoU    │
  │  Recall               Ultralytics val() + manual IoU    │
  │  F1-Score             Computed from P/R                 │
  │  mAP@0.50             Ultralytics val()                 │
  │  mAP@0.50:0.95        Ultralytics val()                 │
  │  Confusion matrix     Per-tile TP/FP/FN                 │
  │  PR curve             Swept confidence thresholds       │
  │  Detection vis.       Overlay GT + predictions          │
  │  Size analysis        XS/S/M/L/XL performance buckets   │
  │  Failure case report  FP / FN analysis                  │
  └─────────────────────────────────────────────────────────┘

Usage:
    python scripts/evaluate.py --weights results/training/oil_tank_yolov8/weights/best.pt
    python scripts/evaluate.py --weights best.pt --split test --conf 0.25
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def iou(a: list[float], b: list[float]) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-9)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> list[float]:
    return [
        (cx - w/2) * img_w, (cy - h/2) * img_h,
        (cx + w/2) * img_w, (cy + h/2) * img_h,
    ]


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Load YOLO label file → list of {box, diameter_px}."""
    results = []
    if not label_path.exists():
        return results
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        box = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)
        diameter = (bw * img_w + bh * img_h) / 2
        results.append({"box": box, "diameter_px": diameter})
    return results


def size_bucket(diameter_px: float) -> str:
    if diameter_px < 20:  return "XS (<20px)"
    if diameter_px < 40:  return "S (20-40px)"
    if diameter_px < 70:  return "M (40-70px)"
    if diameter_px < 110: return "L (70-110px)"
    return "XL (>110px)"


# ─────────────────────────────────────────────────────────────────────────────
# Core matching — per-image TP/FP/FN at a given IoU threshold
# ─────────────────────────────────────────────────────────────────────────────
def match_boxes(
    gt_boxes:   list[dict],
    pred_boxes: list[dict],   # [{box, conf}]
    iou_thresh: float = 0.5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Match predictions to ground truths using greedy IoU matching
    (highest-confidence predictions first).

    Returns: (tp_list, fp_list, fn_list)
    Each element keeps the original dict for downstream analysis.
    """
    preds_sorted = sorted(pred_boxes, key=lambda x: x["conf"], reverse=True)
    matched_gt   = set()
    tps, fps = [], []

    for pred in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou(pred["box"], gt["box"])
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt.add(best_j)
            tps.append({**pred, "matched_gt": gt_boxes[best_j], "iou": best_iou})
        else:
            fps.append({**pred, "iou": best_iou})

    fns = [gt_boxes[j] for j in range(len(gt_boxes)) if j not in matched_gt]
    return tps, fps, fns


# ─────────────────────────────────────────────────────────────────────────────
# PR curve swept over confidence thresholds
# ─────────────────────────────────────────────────────────────────────────────
def compute_pr_curve(
    all_gt:   list[list[dict]],
    all_pred: list[list[dict]],
    iou_thresh: float = 0.5,
    n_thresh: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (thresholds, precisions, recalls) arrays.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresh)
    precisions, recalls = [], []

    for t in thresholds:
        tp = fp = fn = 0
        for gt_list, pred_list in zip(all_gt, all_pred):
            filtered = [p for p in pred_list if p["conf"] >= t]
            t_list, f_list, fn_list = match_boxes(gt_list, filtered, iou_thresh)
            tp += len(t_list); fp += len(f_list); fn += len(fn_list)
        precisions.append(tp / (tp + fp + 1e-9))
        recalls.append(   tp / (tp + fn + 1e-9))

    return thresholds, np.array(precisions), np.array(recalls)


# ─────────────────────────────────────────────────────────────────────────────
# Ultralytics built-in val (mAP etc.)
# ─────────────────────────────────────────────────────────────────────────────
def run_ultralytics_val(
    model, dataset_yaml: Path, split: str, conf: float, iou_thresh: float,
    output_dir: Path
) -> dict:
    val_results = model.val(
        data    = str(dataset_yaml),
        split   = split,
        conf    = conf,
        iou     = iou_thresh,
        imgsz   = 640,
        plots   = True,
        project = str(output_dir / "ultralytics_val"),
        name    = split,
        verbose = False,
    )
    return {
        "precision": float(val_results.box.mp),
        "recall":    float(val_results.box.mr),
        "mAP_50":    float(val_results.box.map50),
        "mAP_50_95": float(val_results.box.map),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#00d4ff"
GREEN    = "#00e676"
RED      = "#ff4d6d"
YELLOW   = "#ffd700"
TEXT_COL = "#e0e0e0"
GRID_COL = "#21262d"


def _style_ax(ax, title: str) -> None:
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=7)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)


def plot_pr_curve(thresholds, precisions, recalls, auc: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Plot P and R vs threshold
    ax.plot(thresholds, precisions, color=ACCENT, lw=2.0, label="Precision")
    ax.plot(thresholds, recalls,    color=GREEN,  lw=2.0, label="Recall",    linestyle="--")
    ax.plot(thresholds, 2*precisions*recalls/(precisions+recalls+1e-9),
            color=YELLOW, lw=1.5, label="F1", linestyle=":")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.text(0.97, 0.06, f"AUC-PR = {auc:.3f}", transform=ax.transAxes,
            ha="right", color=ACCENT, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    _style_ax(ax, "Precision-Recall vs Confidence Threshold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def plot_confusion_matrix(tp: int, fp: int, fn: int, out_path: Path) -> None:
    # 2×2 matrix: rows = GT, cols = Pred
    # [TP, FN]
    # [FP, TN≈0]
    matrix = np.array([[tp, fn], [fp, 0]])
    labels = [["TP", "FN"], ["FP", "TN"]]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            col = "white" if val > matrix.max() * 0.5 else "#ccc"
            ax.text(j, i, f"{labels[i][j]}\n{val:,}", ha="center", va="center",
                    color=col, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Tank", "Predicted: Background"], color=TEXT_COL)
    ax.set_yticklabels(["GT: Tank", "GT: Background"], color=TEXT_COL)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title("Confusion Matrix — Test Set", color=TEXT_COL, fontsize=11, fontweight="bold")
    ax.tick_params(colors=TEXT_COL)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def plot_size_analysis(size_stats: dict, out_path: Path) -> None:
    buckets   = list(size_stats.keys())
    map_vals  = [size_stats[b]["mAP50"] for b in buckets]
    rec_vals  = [size_stats[b]["recall"] for b in buckets]
    count_vals= [size_stats[b]["gt_count"] for b in buckets]
    colors    = [RED, YELLOW, ACCENT, GREEN, "#ff9f43"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)
    ax1, ax2 = axes

    # mAP by size
    bars = ax1.bar(buckets, map_vals, color=colors[:len(buckets)], width=0.6, edgecolor="none")
    for bar, v in zip(bars, map_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                 ha="center", color=TEXT_COL, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.08); ax1.set_xlabel("Tank Size")
    ax1.set_xticklabels(buckets, rotation=15, ha="right")
    _style_ax(ax1, "mAP@0.50 by Tank Size")

    # Count + recall
    x = np.arange(len(buckets))
    ax2b = ax2.twinx()
    ax2.bar(x, count_vals, width=0.4, color=ACCENT, alpha=0.7, label="GT count")
    ax2b.plot(x, rec_vals, color=YELLOW, marker="o", lw=2, label="Recall")
    ax2.set_xticks(x); ax2.set_xticklabels(buckets, rotation=15, ha="right")
    ax2.set_xlabel("Tank Size"); ax2.set_ylabel("GT Box Count", color=TEXT_COL)
    ax2b.set_ylabel("Recall", color=YELLOW)
    ax2b.set_ylim(0, 1.1); ax2b.tick_params(colors=YELLOW)
    ax2.legend(loc="upper left", fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    ax2b.legend(loc="upper right", fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    _style_ax(ax2, "Tank Count & Recall by Size")
    ax2b.set_facecolor(PANEL_BG)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def visualize_detections(
    img_path: Path, gt: list[dict], preds: list[dict],
    tps: list, fps: list, fns: list, out_path: Path
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.imshow(img_rgb)

    def _rect(box, color, lw=2, linestyle="-", label=None):
        x1, y1, x2, y2 = box
        r = mpatches.FancyBboxPatch(
            (x1, y1), x2-x1, y2-y1,
            boxstyle="square,pad=0", lw=lw, edgecolor=color,
            facecolor="none", linestyle=linestyle
        )
        ax.add_patch(r)

    # GT (dashed green)
    for g in gt:
        _rect(g["box"], GREEN, lw=1.5, linestyle="--")

    # TP (solid cyan)
    for t in tps:
        _rect(t["box"], ACCENT, lw=2)
        ax.text(t["box"][0], t["box"][1]-5, f"{t['conf']:.2f}",
                color=ACCENT, fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=DARK_BG, alpha=0.75, edgecolor="none"))

    # FP (red)
    for f in fps:
        _rect(f["box"], RED, lw=2)
        ax.text(f["box"][0], f["box"][1]-5, f"FP {f['conf']:.2f}",
                color=RED, fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=DARK_BG, alpha=0.75, edgecolor="none"))

    # FN (yellow)
    for f in fns:
        _rect(f["box"], YELLOW, lw=1.5, linestyle=":")

    handles = [
        mpatches.Patch(color=GREEN,  label=f"Ground Truth ({len(gt)})", fill=False, linewidth=1.5),
        mpatches.Patch(color=ACCENT, label=f"True Positive ({len(tps)})"),
        mpatches.Patch(color=RED,    label=f"False Positive ({len(fps)})"),
        mpatches.Patch(color=YELLOW, label=f"False Negative ({len(fns)})", fill=False, linestyle="dotted"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              facecolor=DARK_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    ax.set_title(img_path.name, color=TEXT_COL, fontsize=9, pad=4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def plot_failure_analysis(fps: list, fns: list, out_path: Path) -> None:
    """
    Analyse false positives and false negatives.
    FP analysis: distribution of FP confidence scores.
    FN analysis: distribution of missed tank sizes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
    ax1, ax2 = axes

    fp_confs = [f["conf"] for f in fps] or [0]
    fn_diams = [f["diameter_px"] for f in fns] or [0]

    ax1.hist(fp_confs, bins=25, color=RED, edgecolor=DARK_BG, alpha=0.85)
    ax1.axvline(np.mean(fp_confs), color=YELLOW, lw=2, linestyle="--",
                label=f"Mean conf = {np.mean(fp_confs):.2f}")
    ax1.set_xlabel("Confidence Score"); ax1.set_ylabel("Count")
    ax1.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    _style_ax(ax1, f"False Positive Confidence Distribution (n={len(fps)})")

    ax2.hist(fn_diams, bins=25, color=YELLOW, edgecolor=DARK_BG, alpha=0.85)
    ax2.axvline(np.mean(fn_diams), color=RED, lw=2, linestyle="--",
                label=f"Mean size = {np.mean(fn_diams):.1f} px")
    ax2.set_xlabel("Tank Diameter (px)"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    _style_ax(ax2, f"False Negative (Missed Tank) Size Distribution (n={len(fns)})")

    fig.suptitle("Failure Case Analysis — Test Set", color=TEXT_COL, fontsize=12,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    weights:        Path,
    data_dir:       Path,
    split:          str   = "test",
    conf_thresh:    float = 0.25,
    iou_thresh:     float = 0.50,
    output_dir:     Path  = Path("results/eval"),
    max_vis:        int   = 12,
    dataset_yaml:   Path  = Path("configs/_runtime_dataset.yaml"),
) -> dict:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "detection_visualizations"
    vis_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Loading weights: {weights}")
    model = YOLO(str(weights))

    # ── 1. Ultralytics val() for official mAP ────────────────────────────────
    if not dataset_yaml.exists():
        from train import make_dataset_yaml
        make_dataset_yaml(data_dir, dataset_yaml)

    print(f"[INFO] Running Ultralytics val on [{split}] split …")
    ul_metrics = run_ultralytics_val(
        model, dataset_yaml, split, conf_thresh, iou_thresh, output_dir
    )

    # ── 2. Manual per-image matching ─────────────────────────────────────────
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    print(f"[INFO] Manual matching on {len(img_paths)} test images …")

    all_gt_lists:   list[list[dict]] = []
    all_pred_lists: list[list[dict]] = []
    all_tps, all_fps, all_fns = [], [], []

    size_tp:  dict[str, int] = {}
    size_fn:  dict[str, int] = {}
    size_gt:  dict[str, int] = {}

    for img_path in tqdm(img_paths, unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_list = load_gt_boxes(lbl_dir / (img_path.stem + ".txt"), w, h)
        res     = model.predict(str(img_path), conf=conf_thresh, verbose=False)
        pred_list: list[dict] = []
        if res and res[0].boxes is not None:
            for box, cs in zip(res[0].boxes.xyxy.cpu().numpy(),
                               res[0].boxes.conf.cpu().numpy()):
                pred_list.append({"box": box.tolist(), "conf": float(cs)})

        tps, fps, fns = match_boxes(gt_list, pred_list, iou_thresh)
        all_gt_lists.append(gt_list)
        all_pred_lists.append(pred_list)
        all_tps.extend(tps); all_fps.extend(fps); all_fns.extend(fns)

        # Size-stratified stats
        for gt in gt_list:
            b = size_bucket(gt["diameter_px"])
            size_gt[b] = size_gt.get(b, 0) + 1
        for tp in tps:
            b = size_bucket(tp["matched_gt"]["diameter_px"])
            size_tp[b] = size_tp.get(b, 0) + 1
        for fn in fns:
            b = size_bucket(fn["diameter_px"])
            size_fn[b] = size_fn.get(b, 0) + 1

    # ── 3. Aggregate metrics ─────────────────────────────────────────────────
    tp_n = len(all_tps); fp_n = len(all_fps); fn_n = len(all_fns)
    prec = tp_n / (tp_n + fp_n + 1e-9)
    rec  = tp_n / (tp_n + fn_n + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    # ── 4. PR curve ──────────────────────────────────────────────────────────
    thr, precs, recs = compute_pr_curve(all_gt_lists, all_pred_lists, iou_thresh)
    auc = float(np.trapezoid(precs[::-1], recs[::-1]))

    # ── 5. Size-stratified mAP approximation ─────────────────────────────────
    BUCKET_ORDER = ["XS (<20px)", "S (20-40px)", "M (40-70px)", "L (70-110px)", "XL (>110px)"]
    size_stats = {}
    for b in BUCKET_ORDER:
        gt_c = size_gt.get(b, 0)
        tp_c = size_tp.get(b, 0)
        fn_c = size_fn.get(b, 0)
        recall_b  = tp_c / (tp_c + fn_c + 1e-9) if (tp_c + fn_c) > 0 else 0.0
        # Approximate mAP@50 ≈ precision × recall / (1 + ε) (proxy)
        all_tp_b  = tp_c
        fp_b      = sum(1 for f in all_fps
                        if size_bucket(min(f["box"][2]-f["box"][0],
                                          f["box"][3]-f["box"][1])) == b)
        prec_b    = all_tp_b / (all_tp_b + fp_b + 1e-9) if (all_tp_b + fp_b) > 0 else 0.0
        map50_b   = prec_b * recall_b   # proxy — true mAP needs full sweep
        size_stats[b] = {"gt_count": gt_c, "recall": round(recall_b, 3),
                         "mAP50": round(map50_b, 3)}

    # ── 6. Print results ─────────────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  EVALUATION RESULTS — {split.upper()} SET")
    print(f"{'━'*60}")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Precision (manual)':<28} {prec:>10.4f}")
    print(f"  {'Recall (manual)':<28} {rec:>10.4f}")
    print(f"  {'F1-Score':<28} {f1:>10.4f}")
    print(f"  {'mAP@0.50 (Ultralytics)':<28} {ul_metrics['mAP_50']:>10.4f}")
    print(f"  {'mAP@0.50:0.95 (Ultralytics)':<28} {ul_metrics['mAP_50_95']:>10.4f}")
    print(f"  {'AUC-PR':<28} {auc:>10.4f}")
    print(f"  {'TP / FP / FN':<28} {tp_n:>4} / {fp_n:>4} / {fn_n:>4}")
    print(f"{'━'*60}")
    print("\n  Size-stratified mAP@50:")
    for b in BUCKET_ORDER:
        s = size_stats.get(b, {})
        print(f"    {b:<18} mAP≈{s.get('mAP50',0):.3f}  recall={s.get('recall',0):.3f}  n={s.get('gt_count',0)}")
    print()

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    plot_pr_curve(thr, precs, recs, auc, output_dir / "pr_curve.png")
    plot_confusion_matrix(tp_n, fp_n, fn_n,         output_dir / "confusion_matrix.png")
    plot_size_analysis(size_stats,                   output_dir / "size_analysis.png")
    plot_failure_analysis(all_fps, all_fns,          output_dir / "failure_analysis.png")

    # ── 8. Detection visualizations ──────────────────────────────────────────
    sample_paths = img_paths[:max_vis]
    print(f"[INFO] Saving {len(sample_paths)} detection visualizations …")
    for img_path in tqdm(sample_paths, unit="img"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        gt_list   = load_gt_boxes(lbl_dir / (img_path.stem + ".txt"), w, h)
        res       = model.predict(str(img_path), conf=conf_thresh, verbose=False)
        pred_list = []
        if res and res[0].boxes is not None:
            for box, cs in zip(res[0].boxes.xyxy.cpu().numpy(),
                               res[0].boxes.conf.cpu().numpy()):
                pred_list.append({"box": box.tolist(), "conf": float(cs)})
        tps, fps, fns = match_boxes(gt_list, pred_list, iou_thresh)
        visualize_detections(
            img_path, gt_list, pred_list, tps, fps, fns,
            vis_dir / f"{img_path.stem}_eval.jpg"
        )

    # ── 9. Save summary JSON ─────────────────────────────────────────────────
    summary = {
        "split":          split,
        "weights":        str(weights),
        "conf_threshold": conf_thresh,
        "iou_threshold":  iou_thresh,
        "counts":         {"TP": tp_n, "FP": fp_n, "FN": fn_n},
        "metrics": {
            "precision":  round(prec, 4),
            "recall":     round(rec,  4),
            "f1":         round(f1,   4),
            "mAP_50":     round(ul_metrics["mAP_50"],    4),
            "mAP_50_95":  round(ul_metrics["mAP_50_95"], 4),
            "AUC_PR":     round(auc, 4),
        },
        "size_stratified": size_stats,
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Summary saved → {summary_path}")
    print(f"[INFO] Plots    saved → {output_dir}/")
    print(f"[INFO] Visuals  saved → {vis_dir}/")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate oil tank detection model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --weights results/training/oil_tank_yolov8/weights/best.pt
  python scripts/evaluate.py --weights best.pt --split test --conf 0.25 --iou 0.5
        """
    )
    parser.add_argument("--weights",      required=True)
    parser.add_argument("--data-dir",     default="data/processed")
    parser.add_argument("--split",        default="test", choices=["test", "val"])
    parser.add_argument("--conf",         type=float, default=0.25)
    parser.add_argument("--iou",          type=float, default=0.50)
    parser.add_argument("--output-dir",   default="results/eval")
    parser.add_argument("--max-vis",      type=int, default=12)
    parser.add_argument("--dataset-yaml", default="configs/_runtime_dataset.yaml")
    args = parser.parse_args()

    evaluate(
        weights      = Path(args.weights),
        data_dir     = Path(args.data_dir),
        split        = args.split,
        conf_thresh  = args.conf,
        iou_thresh   = args.iou,
        output_dir   = Path(args.output_dir),
        max_vis      = args.max_vis,
        dataset_yaml = Path(args.dataset_yaml),
    )
