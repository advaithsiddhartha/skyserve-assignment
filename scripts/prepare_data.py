"""
prepare_data.py — Airbus Oil Storage Detection Dataset Pipeline
================================================================
Downloads, validates, preprocesses, tiles, and splits the Airbus
Oil Storage Detection dataset for YOLOv8 training.

Reference approach:
  oil_storage-detector (YOLOv5 + Airbus dataset)
  https://github.com/TheodorEmanuelsson/oil_storage-detector

Key design decisions:
  - Tile size: 640×640 (matches YOLOv8 default input size)
  - Overlap: 64 px (prevents tanks being cut at tile edges)
  - Truncation threshold: 30% (discard bbox if <30% visible)
  - Split: 70/20/10 train/val/test (stratified by annotation density)
  - CLAHE: optional contrast enhancement for low-contrast imagery

Usage:
    # Step 1 — Download (requires Kaggle credentials)
    python scripts/prepare_data.py --download

    # Step 2 — Process already-downloaded data
    python scripts/prepare_data.py --raw-dir data/raw

    # Step 3 — Full pipeline with CLAHE
    python scripts/prepare_data.py --download --clahe

    # Pipeline test only (no Kaggle needed)
    python scripts/prepare_data.py --demo-only
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TILE_W       = 640
TILE_H       = 640
TILE_OVERLAP = 64
TRUNC_PCT    = 0.30   # discard annotation if < 30% of its area is in the tile
MIN_BOX_PX   = 8      # discard boxes smaller than 8×8 px in the tile

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20
TEST_RATIO   = 0.10

KAGGLE_SLUG  = "airbusgeo/airbus-oil-storage-detection-dataset"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download
# ─────────────────────────────────────────────────────────────────────────────
def download_dataset(raw_dir: Path) -> None:
    """
    Download the Airbus dataset via Kaggle API.
    Requires ~/.kaggle/kaggle.json with valid credentials.
    Dataset page: https://www.kaggle.com/datasets/airbusgeo/airbus-oil-storage-detection-dataset
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "\n[ERROR] Kaggle credentials not found at ~/.kaggle/kaggle.json\n"
            "  1. Create a Kaggle account at https://www.kaggle.com\n"
            "  2. Go to Account → API → Create New Token\n"
            "  3. Place kaggle.json in ~/.kaggle/\n"
            "  4. Accept the dataset license at:\n"
            "     https://www.kaggle.com/datasets/airbusgeo/airbus-oil-storage-detection-dataset\n"
        )
        sys.exit(1)

    print(f"[INFO] Downloading {KAGGLE_SLUG} → {raw_dir}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
         "-p", str(raw_dir), "--unzip"],
        capture_output=False
    )
    if result.returncode != 0:
        print("[ERROR] Kaggle download failed. Have you accepted the dataset license on Kaggle?")
        sys.exit(1)

    print("[INFO] Download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Annotation parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_annotations(raw_dir: Path) -> dict[str, list[list[float]]]:
    """
    Parse the Airbus annotation file.
    Supports both GeoJSON (.geojson) and CSV formats.

    Returns:
        dict: image_id → list of [xmin, ymin, xmax, ymax] in pixel coords
              (assumes 2560×2560 source images)
    """
    # Try GeoJSON first
    geojson_candidates = list(raw_dir.glob("*.geojson")) + list(raw_dir.glob("**/*.geojson"))
    csv_candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("**/*.csv"))

    if geojson_candidates:
        ann_file = geojson_candidates[0]
        print(f"[INFO] Parsing GeoJSON annotations: {ann_file}")
        return _parse_geojson(ann_file)

    if csv_candidates:
        ann_file = csv_candidates[0]
        print(f"[INFO] Parsing CSV annotations: {ann_file}")
        return _parse_csv(ann_file)

    raise FileNotFoundError(
        f"No annotation file found in {raw_dir}.\n"
        "Expected a .geojson or .csv file with bounding box annotations."
    )


def _parse_geojson(path: Path) -> dict[str, list[list[float]]]:
    """
    Parse a GeoJSON annotation file.
    The Airbus dataset uses pixel-coordinate polygons stored in GeoJSON.
    Each feature has a 'image_id' property and a polygon geometry.
    """
    with open(path) as f:
        data = json.load(f)

    annotations: dict[str, list[list[float]]] = {}

    for feature in data.get("features", []):
        props    = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        # image_id may be stored as 'image_id' or 'ImageId'
        img_id = str(props.get("image_id") or props.get("ImageId") or "")
        if not img_id:
            continue

        coords = geometry.get("coordinates", [[]])[0]  # outer ring of polygon
        if len(coords) < 3:
            continue

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        annotations.setdefault(img_id, []).append([xmin, ymin, xmax, ymax])

    print(f"[INFO] Parsed {sum(len(v) for v in annotations.values())} boxes "
          f"across {len(annotations)} annotated images.")
    return annotations


def _parse_csv(path: Path) -> dict[str, list[list[float]]]:
    """
    Parse a CSV annotation file.
    Expected columns: ImageId, xmin, ymin, xmax, ymax
    """
    import pandas as pd
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    id_col = next(c for c in df.columns if "image" in c or "id" in c.lower())

    annotations: dict[str, list[list[float]]] = {}
    for _, row in df.iterrows():
        img_id = str(row[id_col]).replace(".jpg", "").replace(".jpeg", "")
        box = [float(row["xmin"]), float(row["ymin"]),
               float(row["xmax"]), float(row["ymax"])]
        if box[2] > box[0] and box[3] > box[1]:
            annotations.setdefault(img_id, []).append(box)

    print(f"[INFO] Parsed {sum(len(v) for v in annotations.values())} boxes "
          f"across {len(annotations)} annotated images.")
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLAHE contrast enhancement (satellite-specific)
# ─────────────────────────────────────────────────────────────────────────────
def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    each channel independently in LAB colour space.

    Rationale: SPOT satellite imagery can have low-contrast regions
    due to atmospheric haze or sensor gain differences. CLAHE improves
    local contrast, making tank edges more distinct without introducing
    colour artefacts.
    """
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    L_eq  = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tiling
# ─────────────────────────────────────────────────────────────────────────────
def compute_tile_origins(full_w: int, full_h: int,
                         tile_w: int, tile_h: int,
                         overlap: int) -> list[tuple[int, int]]:
    """
    Compute top-left (x0, y0) origins for all tiles in a sliding window
    with overlap. Ensures full image coverage including the right/bottom edges.
    """
    stride_x = tile_w - overlap
    stride_y = tile_h - overlap

    xs = list(range(0, full_w - tile_w + 1, stride_x))
    ys = list(range(0, full_h - tile_h + 1, stride_y))

    # Guarantee edge coverage
    if not xs or xs[-1] + tile_w < full_w:
        xs.append(max(0, full_w - tile_w))
    if not ys or ys[-1] + tile_h < full_h:
        ys.append(max(0, full_h - tile_h))

    # Deduplicate while preserving order
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    return [(x, y) for y in ys for x in xs]


def clip_box_to_tile(box_abs: list[float], x0: int, y0: int,
                     tile_w: int, tile_h: int,
                     trunc_pct: float, min_px: int
                     ) -> tuple[float, float, float, float] | None:
    """
    Clip an absolute-pixel bounding box to a tile region.

    Returns (cx_norm, cy_norm, w_norm, h_norm) in YOLO format,
    or None if the clipped box is too small / too truncated.
    """
    bx0, by0, bx1, by1 = box_abs

    ix0 = max(bx0, x0);  iy0 = max(by0, y0)
    ix1 = min(bx1, x0 + tile_w);  iy1 = min(by1, y0 + tile_h)

    if ix1 <= ix0 or iy1 <= iy0:
        return None

    inter_area = (ix1 - ix0) * (iy1 - iy0)
    box_area   = max((bx1 - bx0) * (by1 - by0), 1e-6)

    if inter_area / box_area < trunc_pct:
        return None

    # In-tile pixel dimensions
    w_px = ix1 - ix0
    h_px = iy1 - iy0
    if w_px < min_px or h_px < min_px:
        return None

    cx_norm = ((ix0 + ix1) / 2 - x0) / tile_w
    cy_norm = ((iy0 + iy1) / 2 - y0) / tile_h
    w_norm  = w_px / tile_w
    h_norm  = h_px / tile_h

    return cx_norm, cy_norm, w_norm, h_norm


def tile_single_image(
    img_path: Path,
    boxes_abs: list[list[float]],
    out_img_dir: Path,
    out_lbl_dir: Path,
    apply_clahe_flag: bool = False,
    tile_w: int = TILE_W,
    tile_h: int = TILE_H,
    overlap: int = TILE_OVERLAP,
    trunc_pct: float = TRUNC_PCT,
    min_px: int = MIN_BOX_PX,
) -> int:
    """
    Tile one satellite image and write:
      - JPEG tile images to out_img_dir
      - YOLO-format label .txt files to out_lbl_dir

    Returns number of tiles produced.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[WARN] Cannot read {img_path}, skipping.")
        return 0

    if apply_clahe_flag:
        img_bgr = apply_clahe(img_bgr)

    full_h, full_w = img_bgr.shape[:2]
    origins = compute_tile_origins(full_w, full_h, tile_w, tile_h, overlap)

    stem       = img_path.stem
    tile_count = 0

    for x0, y0 in origins:
        tile_bgr = img_bgr[y0: y0 + tile_h, x0: x0 + tile_w]

        yolo_boxes = []
        for box in boxes_abs:
            result = clip_box_to_tile(box, x0, y0, tile_w, tile_h, trunc_pct, min_px)
            if result is not None:
                yolo_boxes.append(result)

        tile_name = f"{stem}_{x0:04d}_{y0:04d}"
        cv2.imwrite(str(out_img_dir / f"{tile_name}.jpg"), tile_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
            for cx, cy, w, h in yolo_boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        tile_count += 1

    return tile_count


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset split (stratified by annotation density)
# ─────────────────────────────────────────────────────────────────────────────
def stratified_split(
    all_images: list[Path],
    annotations: dict[str, list[list[float]]],
    train_r: float = TRAIN_RATIO,
    val_r:   float = VAL_RATIO,
    seed:    int   = 42,
) -> dict[str, list[Path]]:
    """
    Stratified split by annotation density bucket so each split has
    a representative mix of dense, medium, and sparse scenes.
    """
    rng = random.Random(seed)

    def density_bucket(img: Path) -> str:
        n = len(annotations.get(img.stem, annotations.get(img.name, [])))
        if n == 0:   return "empty"
        if n < 50:   return "sparse"
        if n < 150:  return "medium"
        return "dense"

    buckets: dict[str, list[Path]] = {}
    for img in all_images:
        b = density_bucket(img)
        buckets.setdefault(b, []).append(img)

    train, val, test = [], [], []

    for bucket_imgs in buckets.values():
        rng.shuffle(bucket_imgs)
        n     = len(bucket_imgs)
        n_tr  = max(1, round(n * train_r))
        n_val = max(1, round(n * val_r))

        train.extend(bucket_imgs[:n_tr])
        val.extend(bucket_imgs[n_tr: n_tr + n_val])
        test.extend(bucket_imgs[n_tr + n_val:])

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    apply_clahe_flag: bool = False,
    seed: int = 42,
) -> dict[str, int]:
    """Full preprocessing pipeline. Returns tile counts per split."""

    img_dir = raw_dir / "images"
    if not img_dir.exists():
        # Some Kaggle downloads put images directly in raw_dir
        img_dir = raw_dir
    if not any(img_dir.glob("*.jpg")):
        raise FileNotFoundError(
            f"No .jpg images found in {img_dir}.\n"
            "Run with --download first, or place images manually."
        )

    annotations = parse_annotations(raw_dir)

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    print(f"[INFO] Found {len(all_images)} source images.")

    splits = stratified_split(all_images, annotations, seed=seed)
    for s, imgs in splits.items():
        print(f"[INFO]   {s:6s}: {len(imgs):3d} images")

    counts = {}
    for split, imgs in splits.items():
        out_img = processed_dir / "images" / split
        out_lbl = processed_dir / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        total_tiles = 0
        for img_path in tqdm(imgs, desc=f"Tiling [{split}]", unit="img"):
            boxes = annotations.get(img_path.stem,
                    annotations.get(img_path.name, []))
            total_tiles += tile_single_image(
                img_path, boxes, out_img, out_lbl,
                apply_clahe_flag=apply_clahe_flag
            )

        counts[split] = total_tiles
        print(f"[INFO]   {split}: {total_tiles} tiles written.")

    print("\n[INFO] Preprocessing complete.")
    print(f"[INFO]   Train tiles : {counts.get('train', 0)}")
    print(f"[INFO]   Val tiles   : {counts.get('val', 0)}")
    print(f"[INFO]   Test tiles  : {counts.get('test', 0)}")
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 7. Demo mode — pipeline smoke-test without real data
# ─────────────────────────────────────────────────────────────────────────────
def run_demo_pipeline(processed_dir: Path, seed: int = 42) -> None:
    """
    Generate a small set of synthetic tiles to validate that the full
    pipeline (training → evaluation → inference) is functional.

    ⚠️  These images are NOT used for final training or evaluation.
        They exist purely to test that the code runs end-to-end.
    """
    print("\n" + "="*60)
    print("  DEMO MODE — pipeline smoke-test only")
    print("  Synthetic data will NOT be used for real training.")
    print("="*60 + "\n")

    from _synthetic import generate_synthetic_tile   # local helper

    rng = np.random.default_rng(seed)

    counts = {"train": 80, "val": 20, "test": 15}
    for split, n in counts.items():
        out_img = processed_dir / "images" / split
        out_lbl = processed_dir / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n), desc=f"Synth [{split}]"):
            tile_seed = seed * 1000 + {"train": 0, "val": 100000, "test": 200000}[split] + i
            img_arr, boxes = generate_synthetic_tile(seed=tile_seed)
            img = Image.fromarray(img_arr)
            name = f"synth_{split}_{i:04d}"
            img.save(str(out_img / f"{name}.jpg"), "JPEG", quality=88)
            with open(out_lbl / f"{name}.txt", "w") as f:
                for cls, cx, cy, bw, bh in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print("[INFO] Demo tiles created. Proceed with: python scripts/train.py --demo")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Dataset statistics report
# ─────────────────────────────────────────────────────────────────────────────
def print_dataset_stats(processed_dir: Path) -> None:
    print("\n── Dataset Statistics ───────────────────────────────────────")
    for split in ["train", "val", "test"]:
        lbl_dir = processed_dir / "labels" / split
        if not lbl_dir.exists():
            continue
        files     = list(lbl_dir.glob("*.txt"))
        box_sizes = []
        total_boxes = 0
        for f in files:
            lines = f.read_text().strip().split("\n")
            lines = [l for l in lines if l]
            total_boxes += len(lines)
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    w_px = float(parts[3]) * TILE_W
                    h_px = float(parts[4]) * TILE_H
                    box_sizes.append((w_px + h_px) / 2)

        if box_sizes:
            arr = np.array(box_sizes)
            print(f"  {split:6s}: {len(files):5d} tiles | {total_boxes:6d} boxes | "
                  f"size min={arr.min():.0f} mean={arr.mean():.0f} max={arr.max():.0f} px")
        else:
            print(f"  {split:6s}: {len(files):5d} tiles | {total_boxes:6d} boxes")
    print("─────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Airbus Oil Storage Detection dataset for YOLOv8 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download + full pipeline
  python scripts/prepare_data.py --download

  # Process already-downloaded data
  python scripts/prepare_data.py --raw-dir data/raw

  # With CLAHE contrast enhancement
  python scripts/prepare_data.py --download --clahe

  # Pipeline smoke-test (no Kaggle needed)
  python scripts/prepare_data.py --demo-only
        """
    )
    parser.add_argument("--download",      action="store_true",
                        help="Download dataset from Kaggle (requires ~/.kaggle/kaggle.json)")
    parser.add_argument("--raw-dir",       default="data/raw",
                        help="Directory containing downloaded raw images + annotations")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Output directory for tiled dataset")
    parser.add_argument("--clahe",         action="store_true",
                        help="Apply CLAHE contrast enhancement to each tile")
    parser.add_argument("--demo-only",     action="store_true",
                        help="Generate synthetic data for pipeline testing (no Kaggle)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    raw_dir       = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    if args.demo_only:
        sys.path.insert(0, str(Path(__file__).parent))
        run_demo_pipeline(processed_dir, seed=args.seed)
    else:
        if args.download:
            download_dataset(raw_dir)
        run_pipeline(raw_dir, processed_dir,
                     apply_clahe_flag=args.clahe, seed=args.seed)

    print_dataset_stats(processed_dir)
