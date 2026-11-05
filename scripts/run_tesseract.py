"""Run Tesseract (default English LSTM model, automatic page layout) on manifest pages.

Each photo is turned upright from its EXIF flag first, the same as for every other tool.
Tesseract is deterministic, so one run is enough.

Output: runs/tesseract[-<tag>]/<id>.txt
Run (from the repo root, needs Pillow and the tesseract binary; published run: Tesseract 5.5.1):
  python scripts/run_tesseract.py                     # runs/tesseract (reported setting)
  python scripts/run_tesseract.py --psm 11 --tag psm11 # sparse-text mode, kept as a check
"""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(row, out_dir, psm):
    img = ImageOps.exif_transpose(Image.open(os.path.join(BASE, row["image"]))).convert("RGB")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "page.png")
        img.save(path)
        proc = subprocess.run(
            ["tesseract", path, "stdout", "-l", "eng", "--oem", "1", "--psm", str(psm)],
            capture_output=True, text=True, timeout=300,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    open(os.path.join(out_dir, row["id"] + ".txt"), "w", encoding="utf-8").write(proc.stdout)
    return len(proc.stdout.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psm", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "tesseract" + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)

    def job(r):
        try:
            return f"{r['id']} ok {run_one(r, out_dir, a.psm)} words"
        except Exception as e:
            return f"{r['id']} FAILED {str(e)[:200]}"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, msg in enumerate(pool.map(job, rows), 1):
            print(f"[{i}/{len(rows)}]", msg, flush=True)


if __name__ == "__main__":
    main()
