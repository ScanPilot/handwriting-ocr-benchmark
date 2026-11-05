"""Run PaddleOCR 3.x (PP-OCRv5 pipeline) on manifest pages.

Each photo is turned upright from its EXIF flag first, like every tool. Models used in the
published run: PP-OCRv5_mobile_det (text detection), PP-OCRv5_server_rec (recognition, which
PaddleOCR documents as supporting handwriting) and PP-LCNet_x1_0_textline_ori (line
orientation). Recognised lines are written one per line in PaddleOCR's order. Deterministic,
one run. --lang en switches to the English-only recognition model as a check.

Why the mobile detector: with the default server detector, a 4,000 px photo needed ~47 GB of
memory on the CPU build for macOS; capped at 2,560 px it still needed ~21 GB and ~2 minutes a
page. The mobile detector needs ~4 GB, ~6 s a page, and scored higher on our test page.

Output: runs/paddleocr[-<tag>]/<id>.txt
Run (from the repo root, its own environment: pip install paddlepaddle==3.3.1 paddleocr==3.4.1
pillow; published run: PaddleOCR 3.4.1, PaddlePaddle 3.3.1, CPU):
  python scripts/run_paddleocr.py [--det-model PP-OCRv5_server_det --tag server] [--ids ...]
"""
import argparse
import json
import os
import time

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--det-side", type=int, default=2560)
    ap.add_argument("--det-model", default="PP-OCRv5_mobile_det",
                    help="'' for PaddleOCR's default (server) detector")
    a = ap.parse_args()
    # Pages are already upright, so the two pre-processing models (page-orientation classifier
    # and UVDoc unwarping) are switched off: on 4,000 px photos they exhausted 16 GB RAM.
    # Text detection and recognition stay at PaddleOCR's defaults.
    # Detector input capped at 2,560 px on the long side, the same as EasyOCR's default canvas.
    opts = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                text_det_limit_type="max", text_det_limit_side_len=a.det_side)
    if a.det_model:
        opts["text_detection_model_name"] = a.det_model
    ocr = PaddleOCR(lang=a.lang, **opts) if a.lang else PaddleOCR(**opts)
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "paddleocr" + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".txt"))]
    for i, r in enumerate(todo, 1):
        t = time.time()
        try:
            img = ImageOps.exif_transpose(Image.open(os.path.join(BASE, r["image"]))).convert("RGB")
            # PaddleOCR expects BGR arrays like OpenCV
            res = ocr.predict(np.array(img)[:, :, ::-1])
            lines = [txt for page in res for txt in page["rec_texts"]]
            text = "\n".join(lines)
            open(os.path.join(out_dir, r["id"] + ".txt"), "w", encoding="utf-8").write(text)
            print(f"[{i}/{len(todo)}] {r['id']} ok {len(text.split())} words {time.time()-t:.1f}s", flush=True)
        except Exception as e:  # report and move on; a missing file scores as empty
            print(f"[{i}/{len(todo)}] {r['id']} FAILED {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
