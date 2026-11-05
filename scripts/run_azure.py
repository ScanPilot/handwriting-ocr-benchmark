"""Run Azure Document Intelligence prebuilt-read (v4.0 GA, api 2024-11-30) on manifest pages.

Endpoint and key: environment variables AZURE_DI_ENDPOINT and AZURE_DI_KEY (a Document Intelligence
resource on the Standard S0 tier: the free F0 tier caps files at 4 MB and some photos are 9 MB).
The key is sent as a header, never printed or written to outputs. Photos are turned upright from their EXIF
flag first and sent as JPEG q95, the same bytes as Cloud Vision and Textract. Read takes no
handwriting setting, so one run (default settings).

Output: runs/azure[-<tag>]/<id>.txt (page lines in Azure's reading order) and raw/<id>.json
Run (from the repo root, needs Pillow):
  python scripts/run_azure.py [--ids ...] [--tag t]
"""
import argparse
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "2024-11-30"


def load_env():
    endpoint = os.environ.get("AZURE_DI_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_DI_KEY", "")
    if not endpoint or not key:
        raise SystemExit("Set AZURE_DI_ENDPOINT and AZURE_DI_KEY (Document Intelligence, Standard S0)")
    return endpoint, key


def request(url, key, data=None, ctype=None):
    headers = {"Ocp-Apim-Subscription-Key": key}
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {e.read().decode()[:300]}")


def run_one(row, out_dir, endpoint, key):
    img = ImageOps.exif_transpose(Image.open(os.path.join(BASE, row["image"]))).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    url = f"{endpoint}/documentintelligence/documentModels/prebuilt-read:analyze?api-version={API}"
    resp = request(url, key, buf.getvalue(), "image/jpeg")
    op = resp.headers["Operation-Location"]
    for _ in range(120):
        time.sleep(1.5)
        result = json.load(request(op, key))
        if result["status"] == "succeeded":
            break
        if result["status"] == "failed":
            raise RuntimeError(str(result.get("error"))[:300])
    else:
        raise RuntimeError("timed out waiting for analyze result")
    ar = result["analyzeResult"]
    lines = [ln["content"] for p in ar.get("pages", []) for ln in p.get("lines", [])]
    text = "\n".join(lines)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    json.dump({"modelId": ar.get("modelId"), "apiVersion": ar.get("apiVersion"), "lines": lines,
               "styles": ar.get("styles", [])},
              open(os.path.join(out_dir, "raw", row["id"] + ".json"), "w"), ensure_ascii=False)
    open(os.path.join(out_dir, row["id"] + ".txt"), "w", encoding="utf-8").write(text)
    return len(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    endpoint, key = load_env()
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "azure" + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".txt"))]

    def job(r):
        err = ""
        for _ in (1, 2):
            try:
                return f"{r['id']} ok {run_one(r, out_dir, endpoint, key)} words"
            except Exception as e:  # retry once, then report and move on
                err = str(e)[:200]
                time.sleep(3)
        return f"{r['id']} FAILED {err}"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, msg in enumerate(pool.map(job, todo), 1):
            print(f"[{i}/{len(todo)}]", msg, flush=True)


if __name__ == "__main__":
    main()
