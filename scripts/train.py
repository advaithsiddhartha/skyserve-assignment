"""
train.py — YOLOv8 Training for Oil Storage Tank Detection
==========================================================
Trains a YOLOv8 model on the Airbus Oil Storage Detection dataset.

Approach lineage:
  1. oil_storage-detector [YOLOv5 baseline]
       → TheodorEmanuelsson/oil_storage-detector (Kaggle Airbus dataset)
  2. This implementation [YOLOv8 upgrade]
       → Anchor-free head: better for circular tanks of variable size
       → Decoupled classification/regression: improved mAP
       → Mosaic + copy-paste augmentation: crucial for sparse satellite tiles
       → Native mixed-precision (AMP): ~2× speedup on modern GPUs

Usage:
    # Full training on Airbus data (GPU recommended)
    python scripts/train.py

    # With custom config
    python scripts/train.py --model yolov8m --epochs 100 --batch 16

    # Segmentation variant (bonus)
    python scripts/train.py --task segment --model yolov8m-seg

    # Demo / pipeline test (synthetic data)
    python scripts/train.py --demo
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU: {name}  ({mem:.1f} GB VRAM)")
        return "0"
    if torch.backends.mps.is_available():
        print("[INFO] Apple MPS detected")
        return "mps"
    print("[INFO] No GPU found — using CPU (training will be slow)")
    return "cpu"


def load_train_config(cfg_path: Path) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def make_dataset_yaml(processed_dir: Path, out_path: Path) -> None:
    """Write a runtime dataset YAML with absolute paths (avoids CWD issues)."""
    cfg = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc": 1,
        "names": {0: "oil-storage-tank"},
    }
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ─────────────────────────────────────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────────────────────────────────────
def train(
    processed_dir: Path,
    model_name:    str  = "yolov8m",
    task:          str  = "detect",
    epochs:        int  = 100,
    batch:         int  = 16,
    imgsz:         int  = 640,
    run_name:      str  = "oil_tank_yolov8",
    project:       str  = "results/training",
    patience:      int  = 20,
    seed:          int  = 42,
    resume:        bool = False,
    cfg_path:      Path = Path("configs/train.yaml"),
) -> Path:
    """
    Train YOLOv8 on the Airbus Oil Storage dataset.

    Returns:
        Path to best.pt weights file
    """
    from ultralytics import YOLO

    # ── Validate data exists ─────────────────────────────────────────────────
    train_dir = processed_dir / "images" / "train"
    if not train_dir.exists() or not any(train_dir.glob("*.jpg")):
        raise FileNotFoundError(
            f"Training images not found in {train_dir}.\n"
            "Run: python scripts/prepare_data.py --download"
        )

    # ── Build runtime dataset YAML ───────────────────────────────────────────
    dataset_yaml = Path("configs/_runtime_dataset.yaml")
    make_dataset_yaml(processed_dir, dataset_yaml)

    # ── Load hyperparameter config ───────────────────────────────────────────
    hp = {}
    if cfg_path.exists():
        hp = load_train_config(cfg_path)
        # Allow CLI overrides
        if epochs  != 100: hp["epochs"]  = epochs
        if batch   != 16:  hp["batch"]   = batch
        if imgsz   != 640: hp["imgsz"]   = imgsz
        if patience != 20: hp["patience"] = patience
    else:
        hp = dict(epochs=epochs, batch=batch, imgsz=imgsz, patience=patience,
                  seed=seed, optimizer="AdamW", lr0=0.001, lrf=0.01,
                  weight_decay=0.0005, warmup_epochs=3,
                  hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
                  degrees=90.0, translate=0.1, scale=0.6,
                  flipud=0.5, fliplr=0.5,
                  mosaic=1.0, mixup=0.15, copy_paste=0.15)

    device = detect_device()

    # ── Adapt batch for CPU ──────────────────────────────────────────────────
    effective_batch = hp.get("batch", batch)
    if device == "cpu" and effective_batch > 4:
        effective_batch = 4
        print(f"[WARN] CPU mode — reducing batch to {effective_batch}")

    # ── Model weights ────────────────────────────────────────────────────────
    weights = f"{model_name}.pt"
    if task == "segment":
        weights = f"{model_name}-seg.pt" if "-seg" not in model_name else f"{model_name}.pt"

    print(f"\n{'='*60}")
    print(f"  Model  : {weights}")
    print(f"  Task   : {task}")
    print(f"  Epochs : {hp.get('epochs', epochs)}")
    print(f"  Batch  : {effective_batch}")
    print(f"  ImgSz  : {hp.get('imgsz', imgsz)}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[WARN] Could not load pretrained {weights}: {e}")
        yaml_name = weights.replace(".pt", ".yaml")
        print(f"[INFO] Falling back to training from scratch: {yaml_name}")
        model = YOLO(yaml_name)

    # ── Extract augmentation + loss params from hp ───────────────────────────
    aug_keys = {
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
        "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
        "copy_paste",
    }
    opt_keys = {"optimizer", "lr0", "lrf", "momentum", "weight_decay",
                "warmup_epochs", "warmup_momentum"}
    loss_keys = {"box", "cls", "dfl"}

    extra_hp = {k: v for k, v in hp.items()
                if k in aug_keys | opt_keys | loss_keys}
    extra_hp.pop("model", None)  # not a valid train kwarg

    results = model.train(
        data         = str(dataset_yaml),
        epochs       = hp.get("epochs", epochs),
        batch        = effective_batch,
        imgsz        = hp.get("imgsz", imgsz),
        device       = device,
        project      = project,
        name         = run_name,
        patience     = hp.get("patience", patience),
        seed         = hp.get("seed", seed),
        amp          = hp.get("amp", True) and device != "cpu",
        save         = True,
        save_period  = 10,
        val          = True,
        plots        = True,
        verbose      = True,
        resume       = resume,
        **extra_hp,
    )

    best = Path(project) / run_name / "weights" / "best.pt"
    if best.exists():
        print(f"\n[INFO] Training complete. Best weights → {best}")
    else:
        # Search fallback
        candidates = sorted(Path(project).glob("**/best.pt"))
        best = candidates[-1] if candidates else None
        if best:
            print(f"\n[INFO] Best weights found at: {best}")
        else:
            print("[WARN] Could not locate best.pt")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Demo shortcut
# ─────────────────────────────────────────────────────────────────────────────
def train_demo(model_name="yolov8n", epochs=30) -> Path:
    """
    Run a quick training on synthetic data to validate the pipeline.
    NOT for reporting results — only confirms the code works.
    """
    import subprocess

    processed_dir = Path("data/processed")
    if not (processed_dir / "images" / "train").exists():
        print("[INFO] Generating synthetic demo tiles …")
        subprocess.run(
            [sys.executable, "scripts/prepare_data.py", "--demo-only"],
            check=True
        )

    print("\n[WARN] ─────────────────────────────────────────────────────────")
    print("[WARN]  DEMO MODE: Training on SYNTHETIC data.")
    print("[WARN]  Metrics from this run are NOT valid for reporting.")
    print("[WARN]  Use real Airbus data for final training.")
    print("[WARN] ─────────────────────────────────────────────────────────\n")

    return train(
        processed_dir = processed_dir,
        model_name    = model_name,
        epochs        = epochs,
        batch         = 4,
        run_name      = "oil_tank_demo",
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for oil storage tank detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                            # full training, defaults
  python scripts/train.py --model yolov8l --epochs 150
  python scripts/train.py --task segment --model yolov8m-seg
  python scripts/train.py --demo                    # pipeline smoke-test
        """
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model",   default="yolov8m",
                        help="yolov8n/s/m/l/x or yolov8m-seg for segmentation")
    parser.add_argument("--task",    default="detect", choices=["detect", "segment"])
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--name",    default="oil_tank_yolov8")
    parser.add_argument("--patience",type=int, default=20)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--demo",    action="store_true",
                        help="Train on synthetic data for pipeline testing only")
    args = parser.parse_args()

    if args.demo:
        train_demo(model_name=args.model, epochs=min(args.epochs, 30))
    else:
        train(
            processed_dir = Path(args.processed_dir),
            model_name    = args.model,
            task          = args.task,
            epochs        = args.epochs,
            batch         = args.batch,
            imgsz         = args.imgsz,
            run_name      = args.name,
            patience      = args.patience,
            seed          = args.seed,
            resume        = args.resume,
        )
