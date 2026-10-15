"""Build the chronological 2x2 composite sent to the VLM."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .types import Frame


def _grid(n: int) -> Tuple[int, int]:
    """(rows, cols) layout; reading order is left-to-right, then top-to-bottom."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    return 2, 2


def build_composite(frames: Sequence[Frame], max_side: int, label: bool = True) -> Tuple[np.ndarray, int, List[float]]:
    """Tile up to four frames chronologically.

    Returns (composite BGR image, actual sample count, frame times). With fewer
    than four frames the available ones are used (no duplication) and the real
    count is returned so it can be recorded. Tiles carry a small ``1..4`` label
    and relative-seconds stamp to reinforce the temporal order for the VLM.
    """
    frames = list(frames)[:4]
    n = len(frames)
    if n == 0:
        raise ValueError("no frames available for composite")
    rows, cols = _grid(n)
    h0, w0 = frames[-1].image.shape[:2]
    tile_w, tile_h = w0, h0
    comp_w, comp_h = tile_w * cols, tile_h * rows
    scale = max_side / max(comp_w, comp_h)
    tile_w = max(16, int(tile_w * scale))
    tile_h = max(16, int(tile_h * scale))
    canvas = np.zeros((tile_h * rows, tile_w * cols, 3), np.uint8)
    t_last = frames[-1].t
    for i, fr in enumerate(frames):
        img = cv2.resize(fr.image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        if label:
            txt = f"{i + 1}  {fr.t - t_last:+.1f}s"
            cv2.rectangle(img, (0, 0), (8 + 9 * len(txt), 20), (0, 0, 0), -1)
            cv2.putText(img, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = img
    return canvas, n, [f.t for f in frames]
