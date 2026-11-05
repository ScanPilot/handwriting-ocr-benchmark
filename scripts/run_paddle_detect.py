"""Text-line detection only (PaddleOCR PP-OCRv5_mobile_det), for line readers such as TrOCR.

TrOCR reads one text line at a time and has no detector of its own, so the TrOCR pipeline is
"PaddleOCR's detector + TrOCR's reader". Same detector settings as the PaddleOCR run (upright
page, long side capped at 2,560 px). Saves the line polygons in the upright image's pixels.

Output: runs/_detect/paddle_mobile/<id>.json  ({"size": [w, h], "polys": [[[x, y] x4], ...]})
The published polygons are in this repo (runs/_detect/paddle_mobile/), so TrOCR can be rerun
without installing PaddleOCR.
Run (from the repo root, same environment as run_paddleocr.py):
  python scripts/run_paddle_detect.py [--ids ...]
"""
import argparse
import json
import os
import time

import numpy as np
from paddleocr import TextDetection
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    det = TextDetection(model_name="PP-OCRv5_mobile_det", limit_side_len=2560, limit_type="max")
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "_detect", "paddle_mobile")
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".json"))]
    for i, r in enumerate(todo, 1):
        t = time.time()
        img = ImageOps.exif_transpose(Image.open(os.path.join(BASE, r["image"]))).convert("RGB")
        res = det.predict(np.array(img)[:, :, ::-1])
        polys = [np.asarray(p).astype(int).tolist() for page in res for p in page["dt_polys"]]
        json.dump({"size": list(img.size), "polys": polys},
                  open(os.path.join(out_dir, r["id"] + ".json"), "w"))
        print(f"[{i}/{len(todo)}] {r['id']} {len(polys)} lines {time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
