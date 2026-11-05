"""Run Google Cloud Vision DOCUMENT_TEXT_DETECTION on manifest pages.

Key: environment variable GOOGLE_VISION_API_KEY (an API key restricted to the Vision API; sent
as a header, never printed or written to outputs). Photos are turned upright from their EXIF flag first, like every tool.
Vision is deterministic enough for one run per setting.

Settings:
  default            no language hint (what most integrations send)
  --hint handwriting languageHints ["en-t-i0-handwrit"], Google's handwriting hint

Output: runs/vision[-<tag>]/<id>.txt and raw/<id>.json
Run (from the repo root, needs Pillow):
  python scripts/run_vision.py                              # runs/vision
  python scripts/run_vision.py --hint handwriting --tag hw  # runs/vision-hw (reported setting)
"""
import argparse
import base64
import io
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://vision.googleapis.com/v1/images:annotate"


def load_key():
    key = os.environ.get("GOOGLE_VISION_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_VISION_API_KEY (a Google Cloud API key with the Vision API enabled)")
    return key


def run_one(row, out_dir, key, hints):
    img = ImageOps.exif_transpose(Image.open(os.path.join(BASE, row["image"]))).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    req_body = {"image": {"content": base64.b64encode(buf.getvalue()).decode()},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}
    if hints:
        req_body["imageContext"] = {"languageHints": hints}
    req = urllib.request.Request(
        URL, data=json.dumps({"requests": [req_body]}).encode(),
        headers={"Content-Type": "application/json", "X-Goog-Api-Key": key})
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=120))["responses"][0]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {e.read().decode()[:300]}")
    if "error" in resp:
        raise RuntimeError(str(resp["error"])[:300])
    text = resp.get("fullTextAnnotation", {}).get("text", "")
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    json.dump({"text": text}, open(os.path.join(out_dir, "raw", row["id"] + ".json"), "w"))
    open(os.path.join(out_dir, row["id"] + ".txt"), "w", encoding="utf-8").write(text)
    return len(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint", choices=["none", "handwriting"], default="none")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    key = load_key()
    hints = ["en-t-i0-handwrit"] if a.hint == "handwriting" else None
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "vision" + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".txt"))]

    def job(r):
        for _ in (1, 2):
            try:
                return f"{r['id']} ok {run_one(r, out_dir, key, hints)} words"
            except Exception as e:
                err = str(e)[:200]
        return f"{r['id']} FAILED {err}"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, msg in enumerate(pool.map(job, todo), 1):
            print(f"[{i}/{len(todo)}]", msg, flush=True)


if __name__ == "__main__":
    main()
