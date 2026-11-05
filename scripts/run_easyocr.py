"""Run EasyOCR (English model, default settings) on manifest pages.

Each photo is turned upright from its EXIF flag first, like every tool. EasyOCR detects text
boxes (CRAFT) and reads each one; boxes are written one per line in EasyOCR's order.
Deterministic, so one run. Defaults include canvas_size 2560 (EasyOCR downsizes larger photos).

Output: runs/easyocr[-<tag>]/<id>.txt
Run (from the repo root, environment from requirements-open-source.txt; published run:
EasyOCR 1.7.2, PyTorch 2.15 on Apple MPS):
  python scripts/run_easyocr.py [--ids ...] [--tag t]
"""
import argparse
import json
import os
import time

import easyocr
import numpy as np
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    reader = easyocr.Reader(["en"], verbose=False)
    print("EasyOCR", easyocr.__version__, "device", reader.device, flush=True)
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "easyocr" + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".txt"))]
    for i, r in enumerate(todo, 1):
        t = time.time()
        try:
            img = ImageOps.exif_transpose(Image.open(os.path.join(BASE, r["image"]))).convert("RGB")
            lines = reader.readtext(np.array(img), detail=0)
            text = "\n".join(lines)
            open(os.path.join(out_dir, r["id"] + ".txt"), "w", encoding="utf-8").write(text)
            print(f"[{i}/{len(todo)}] {r['id']} ok {len(text.split())} words {time.time()-t:.1f}s", flush=True)
        except Exception as e:  # report and move on; a missing file scores as empty
            print(f"[{i}/{len(todo)}] {r['id']} FAILED {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
