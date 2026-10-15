#!/usr/bin/env python
"""Generate SYNTHETIC demo/evaluation clips.

Real people are cut out of the two photos that ship inside the Ultralytics
package (bus.jpg, zidane.jpg) with YOLOv8n and pasted onto a plain floor
background that moves along scripted paths. Every clip is therefore *synthetic*
(not real surveillance footage): it exercises detector -> tracker -> heuristics
-> VLM -> storage end to end and gives deterministic, reproducible inputs, but it
says little about real-world accuracy. Provenance is recorded in eval/manifest.csv.

Usage: python scripts/make_demo_clips.py [--out data/clips] [--normal-minutes 10]
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np

W, H, FPS = 640, 480, 10


def load_sprites() -> List[np.ndarray]:
    import ultralytics
    from ultralytics import YOLO

    assets = Path(ultralytics.__file__).parent / "assets"
    model = YOLO("yolov8n.pt")
    sprites: List[np.ndarray] = []
    for name in ("bus.jpg", "zidane.jpg"):
        img = cv2.imread(str(assets / name))
        res = model.predict(img, conf=0.5, classes=[0], verbose=False, device="cpu")[0]
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy().astype(int):
            crop = img[max(0, y1):y2, max(0, x1):x2].copy()
            if crop.shape[0] > 120 and crop.shape[1] > 40 and crop.shape[0] / crop.shape[1] > 1.6:  # full-ish standing person
                sprites.append(crop)
    if len(sprites) < 2:
        raise SystemExit("could not extract enough person sprites from the Ultralytics sample images")
    return sprites


def background() -> np.ndarray:
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        bg[y, :] = (150 - y // 12, 160 - y // 12, 165 - y // 12)
    for x in range(0, W, 80):
        cv2.line(bg, (x, 0), (x, H), (120, 125, 128), 1)
    for y in range(120, H, 60):
        cv2.line(bg, (0, y), (W, y), (125, 130, 133), 1)
    return bg


def paste(canvas: np.ndarray, sprite: np.ndarray, foot: Tuple[float, float], height: float, angle: float = 0.0,
          flip: bool = False) -> None:
    """Paste ``sprite`` with its feet at ``foot``, rotated by ``angle`` deg around the feet."""
    s = height / sprite.shape[0]
    sp = cv2.resize(sprite, (max(8, int(sprite.shape[1] * s)), int(height)))
    if flip:
        sp = sp[:, ::-1]
    h, w = sp.shape[:2]
    if abs(angle) > 0.5:
        pad = int(max(h, w))
        big = cv2.copyMakeBorder(sp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        M = cv2.getRotationMatrix2D((pad + w / 2, pad + h), angle, 1.0)  # pivot at the feet
        rot = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(np.full(big.shape[:2], 255, np.uint8) * 0 + np.pad(np.full((h, w), 255, np.uint8), pad), M,
                              (big.shape[1], big.shape[0]))
        ys, xs = np.where(mask > 127)
        sp = rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ox = int(foot[0] - (pad + w / 2 - xs.min()))
        oy = int(foot[1] - (pad + h - ys.min()))
    else:
        ox, oy = int(foot[0] - w / 2), int(foot[1] - h)
    h2, w2 = sp.shape[:2]
    x1, y1, x2, y2 = max(0, ox), max(0, oy), min(W, ox + w2), min(H, oy + h2)
    if x2 > x1 and y2 > y1:
        canvas[y1:y2, x1:x2] = sp[y1 - oy:y2 - oy, x1 - ox:x2 - ox]


def write(path: Path, seconds: float, draw: Callable[[np.ndarray, float], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    bg = background()
    for k in range(int(seconds * FPS)):
        frame = bg.copy()
        draw(frame, k / FPS)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        vw.write(frame)
    vw.release()
    print(f"wrote {path} ({seconds:.0f}s)")


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, u))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--normal-minutes", type=float, default=10.0)
    ap.add_argument("--skip-normal", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    S = load_sprites()
    sp = lambda i: S[i % len(S)]  # noqa: E731
    HT = 230.0

    def person_down(f, t):  # walks in, stands, collapses at t=4.0, lies still
        x = lerp(100, 300, (t - 0.5) / 3.0)
        if t < 4.0:
            paste(f, sp(0), (x, 420), HT)
        else:
            ang = lerp(0, -90, (t - 4.0) / 0.6)
            paste(f, sp(0), (300, 420), HT, ang)

    def tailgating(f, t, lag=0.8):  # two people cross the entry line (y=0.55*H=264) `lag` s apart
        for i, (x0, start) in enumerate(((260, 1.0), (330, 1.0 + lag))):
            y = lerp(190, 440, (t - start) / 3.5)
            if t >= start - 0.1:
                paste(f, sp(i), (x0, y), HT * (0.7 + 0.3 * (y - 190) / 250) + 40)

    def suspicious(f, t):  # stands in the restricted zone (x>0.7W) from 2s to 14s
        if 2.0 <= t <= 14.0:
            paste(f, sp(1), (545, 430), HT)

    def crowd(f, t):  # 8 people gather on the left 2/3 of the frame
        if t >= 2.0:
            for i in range(8):
                x = 60 + (i % 4) * 100 + 30 * (i // 4)
                y = 330 + 90 * (i // 4)
                paste(f, sp(i), (x + 3 * math.sin(t * 1.5 + i), y), 170, flip=bool(i % 2))

    def fighting(f, t):  # two people stay close and jostle fast in opposite phase (up/down + lunging)
        if t >= 1.5:
            ph = 2 * math.pi * (t - 1.5) / 0.8
            for i, s in enumerate((1, -1)):
                x = (235 if i == 0 else 385) + s * 10 * math.sin(ph)
                paste(f, sp(0 if i == 0 else 2), (x, 420 + s * 28 * math.sin(ph)), HT + 14 * s * math.sin(ph), flip=(i == 1))

    write(out / "person_down_01.mp4", 12, person_down)
    write(out / "tailgating_01.mp4", 8, lambda f, t: tailgating(f, t, 0.8))
    write(out / "tailgating_negative_01.mp4", 12, lambda f, t: tailgating(f, t, 4.5))
    write(out / "suspicious_01.mp4", 16, suspicious)
    write(out / "crowd_01.mp4", 12, crowd)
    write(out / "fighting_01.mp4", 10, fighting)
    # demo clips referenced by configs/cameras.yaml
    write(out / "demo_cam1.mp4", 24, lambda f, t: (person_down(f, t) if t < 12 else suspicious(f, t - 12 + 2)))
    write(out / "demo_cam2.mp4", 8, lambda f, t: tailgating(f, t, 0.8))

    if a.skip_normal:
        return
    rng = random.Random(7)

    def normal(f, t):  # walkers cross left->right every ~20 s, sometimes two far apart; never stop
        for k in range(int(t // 20) + 1):
            t0 = k * 20.0 + (k * 7) % 5
            u = (t - t0) / 6.0
            if 0 <= u <= 1:
                paste(f, sp(k), (lerp(-40, 440, u), 440), HT, flip=bool(k % 2))
                if k % 3 == 0:
                    paste(f, sp(k + 1), (lerp(-240, 240, u), 440), HT - 20)

    write(out / "eval" / "normal_01.mp4", a.normal_minutes * 60, normal)


if __name__ == "__main__":
    main()
