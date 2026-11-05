"""Run TrOCR (microsoft/trocr-large-handwritten) on lines found by PaddleOCR's detector.

TrOCR is a line-level handwriting reader with no detector, so each page goes through
runs/_detect/paddle_mobile/<id>.json (scripts/run_paddle_detect.py) first. Every detected line
is cropped from the upright page (bounding box of the polygon, 4 px padding), read by TrOCR with
the model's own generation settings, and written top to bottom, left to right.

Output: runs/trocr[-<tag>]/<id>.txt
Run (from the repo root, environment from requirements-open-source.txt; published run:
transformers 5.22.1, PyTorch 2.15 on Apple MPS, batches of 8 lines, max 64 new tokens):
  python scripts/run_trocr.py [--ids ...] [--tag t] [--model ...]
"""
import argparse
import json
import os
import time

import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = os.path.join(BASE, "runs", "_detect", "paddle_mobile")


def boxes(polys, size, pad=4):
    w, h = size
    out = []
    for p in polys:
        xs, ys = [pt[0] for pt in p], [pt[1] for pt in p]
        x0, y0, x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad), min(w, max(xs) + pad), min(h, max(ys) + pad)
        if x1 - x0 > 4 and y1 - y0 > 4:
            out.append((x0, y0, x1, y1))
    # reading order: top to bottom by line centre, then left to right
    return sorted(out, key=lambda b: (round((b[1] + b[3]) / 2 / 20), b[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="microsoft/trocr-large-handwritten")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(a.model)
    model = VisionEncoderDecoderModel.from_pretrained(a.model).to(device).eval()
    print("TrOCR", a.model, "device", device, flush=True)
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "trocr" + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".txt"))]
    for i, r in enumerate(todo, 1):
        t = time.time()
        try:
            det = json.load(open(os.path.join(DET, r["id"] + ".json")))
            img = ImageOps.exif_transpose(Image.open(os.path.join(BASE, r["image"]))).convert("RGB")
            crops = [img.crop(b) for b in boxes(det["polys"], det["size"])]
            lines = []
            for k in range(0, len(crops), a.batch):
                pix = processor(images=crops[k:k + a.batch], return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    ids = model.generate(pix, max_new_tokens=64)
                lines += processor.batch_decode(ids, skip_special_tokens=True)
            text = "\n".join(l.strip() for l in lines if l.strip())
            open(os.path.join(out_dir, r["id"] + ".txt"), "w", encoding="utf-8").write(text)
            print(f"[{i}/{len(todo)}] {r['id']} ok {len(crops)} lines {len(text.split())} words {time.time()-t:.1f}s", flush=True)
        except Exception as e:  # report and move on; a missing file scores as empty
            print(f"[{i}/{len(todo)}] {r['id']} FAILED {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
