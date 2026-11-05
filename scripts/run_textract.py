"""Run Amazon Textract DetectDocumentText (plain OCR, printed + handwriting) on manifest pages.

AWS credentials from the standard chain (AWS_PROFILE, env vars, aws login, ...). Region
eu-central-1 (Frankfurt), the Region used for the published run. Photos are turned upright from their EXIF flag first, like every
tool, and sent as JPEG q95 like Cloud Vision; quality drops only if a page would exceed
Textract's 10 MB request limit (logged per page). Textract is deterministic, so one run.

Output: runs/textract[-<tag>]/<id>.txt (LINE blocks in Textract's reading order) and raw/<id>.json
Run (from the repo root, needs boto3 and Pillow; aws login credentials need botocore[crt]):
  python scripts/run_textract.py [--ids ...] [--tag t] [--profile NAME] [--region eu-central-1]
"""
import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 10 * 1024 * 1024


def jpeg_bytes(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    for q in (95, 90, 85, 80):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        if buf.tell() < LIMIT:
            return buf.getvalue(), q
    raise RuntimeError("page too large even at q80")


def run_one(client, row, out_dir):
    data, q = jpeg_bytes(os.path.join(BASE, row["image"]))
    resp = client.detect_document_text(Document={"Bytes": data})
    lines = [b["Text"] for b in resp.get("Blocks", []) if b["BlockType"] == "LINE"]
    text = "\n".join(lines)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    json.dump({"quality": q, "lines": lines,
               "words": [{"text": b["Text"], "type": b.get("TextType"), "conf": b.get("Confidence")}
                         for b in resp.get("Blocks", []) if b["BlockType"] == "WORD"]},
              open(os.path.join(out_dir, "raw", row["id"] + ".json"), "w"), ensure_ascii=False)
    open(os.path.join(out_dir, row["id"] + ".txt"), "w", encoding="utf-8").write(text)
    return len(text.split()), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="eu-central-1")
    a = ap.parse_args()
    client = boto3.Session(profile_name=a.profile, region_name=a.region).client("textract")
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    out_dir = os.path.join(BASE, "runs", "textract" + (f"-{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".txt"))]

    def job(r):
        err = ""
        for _ in (1, 2):
            try:
                n, q = run_one(client, r, out_dir)
                return f"{r['id']} ok {n} words q{q}"
            except Exception as e:  # retry once, then report and move on
                err = str(e)[:200]
        return f"{r['id']} FAILED {err}"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, msg in enumerate(pool.map(job, todo), 1):
            print(f"[{i}/{len(todo)}]", msg, flush=True)


if __name__ == "__main__":
    main()
