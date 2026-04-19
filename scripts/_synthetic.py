"""
_synthetic.py — Synthetic Tile Generator (pipeline smoke-test ONLY)
====================================================================
⚠️  This module is used EXCLUSIVELY for pipeline validation.
    It is NOT part of training or evaluation on real data.
    Final results must use the Airbus Oil Storage Detection dataset.

Generates 640×640 satellite-like tiles with circular oil tanks,
preserving the visual properties of SPOT imagery (sandy/industrial
backgrounds, silver circular tanks with inner shadow).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a sandy/industrial satellite background."""
    base = np.array([
        rng.integers(130, 178),
        rng.integers(118, 162),
        rng.integers(88, 138),
    ], dtype=np.uint8)

    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        canvas[:, :, c] = base[c] + rng.normal(0, 11, (h, w))

    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    for _ in range(rng.integers(6, 16)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        shade = int(base[0]) + int(rng.integers(-18, 18))
        col = tuple(int(np.clip(shade + d, 0, 255)) for d in (0, -8, -18))
        draw.rectangle(
            [x0, y0, x0 + rng.integers(15, 70), y0 + rng.integers(15, 70)],
            fill=col
        )
    return np.array(img.filter(ImageFilter.GaussianBlur(radius=0.7)))


def _draw_tank(draw: ImageDraw.ImageDraw, cx: int, cy: int,
               r: int, rng: np.random.Generator) -> None:
    """Draw a circular oil storage tank with realistic satellite appearance."""
    # Outer shadow ring
    shadow = tuple(int(rng.integers(35, 65)) for _ in range(3))
    draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2], fill=shadow)

    # Tank body (silver/white-grey)
    b = int(rng.integers(175, 242))
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(b, b, b))

    # Floating roof shadow (darker inner ellipse, offset)
    ir = int(r * 0.52)
    ib = int(rng.integers(75, 135))
    off = int(r * 0.14)
    draw.ellipse([cx-ir+off, cy-ir+off, cx+ir+off, cy+ir+off],
                 fill=(ib, ib, max(0, ib - 8)))

    # Highlight glint (top-left)
    hr = max(2, int(r * 0.18))
    ho = -int(r * 0.38)
    draw.ellipse([cx+ho, cy+ho, cx+ho+hr, cy+ho+hr], fill=(255, 255, 252))


def generate_synthetic_tile(
    w: int = 640, h: int = 640, seed: int | None = None
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """
    Generate one synthetic 640×640 tile.

    Returns:
        (img_array_RGB, yolo_labels)
        yolo_labels: list of (class_id, cx_norm, cy_norm, w_norm, h_norm)
    """
    rng = np.random.default_rng(seed)
    arr = _background(w, h, rng)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    labels = []
    placed = []
    n_tanks = int(rng.integers(0, 9))

    for _ in range(n_tanks):
        r  = int(rng.integers(10, 54))
        m  = r + 5
        cx = int(rng.integers(m, w - m))
        cy = int(rng.integers(m, h - m))

        if any(np.sqrt((cx-px)**2 + (cy-py)**2) < r+pr+4
               for px, py, pr in placed):
            continue

        placed.append((cx, cy, r))
        _draw_tank(draw, cx, cy, r, rng)

        labels.append((
            0,
            cx / w,
            cy / h,
            (2 * r) / w,
            (2 * r) / h,
        ))

    img = img.filter(ImageFilter.SHARPEN)
    return np.array(img), labels
