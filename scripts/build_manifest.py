"""Build data/manifest.jsonl: one row per test page with its reference text.

Sources
- GNHK test split (CC BY 4.0, Lee, Chung & Lee, ICDAR 2021): photos of handwritten notes.
  Reference = handwritten words (type H) in line order, left to right.
  Dropped tokens: %NA% (unreadable), %SC% (crossed out), %math% (symbols).
  Printed words (type P) are not required, but a tool that outputs them is not penalised.
- OmniAI ocr-benchmark (MIT): 11 scanned forms with handwritten fill-ins, picked by eye.
  Reference = their markdown with the markdown syntax removed. All words are required.

Run: python3 -I scripts/build_manifest.py   (from the repo root)
"""
import glob
import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GNHK_DIR = os.path.join(BASE, "data/gnhk-raw/test/test")
OMNI_META = os.path.join(BASE, "data/omni-raw/test/metadata.jsonl")
OMNI_IDS = [880, 661, 46, 36, 40, 54, 426, 43, 26, 430, 41]
DROP = {"%NA%", "%SC%", "%math%"}


def gnhk_rows():
    for path in sorted(glob.glob(os.path.join(GNHK_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        words = json.load(open(path))
        lines = {}
        printed = []
        math_tokens = sum(1 for w in words if w["text"] == "%math%")
        # clean = nothing the reference leaves out (crossed out, unreadable, math)
        clean = not any(w["text"] in DROP for w in words)
        for w in words:
            if w["text"] in DROP:
                continue
            if w["type"] == "P":
                printed.append(w["text"])
                continue
            x = min(w["polygon"][k] for k in ("x0", "x1", "x2", "x3"))
            lines.setdefault(w["line_idx"], []).append((x, w["text"]))
        text = "\n".join(
            " ".join(t for _, t in sorted(lines[i])) for i in sorted(lines)
        )
        yield {
            "id": f"gnhk_{name}",
            "source": "GNHK",
            "category": "handwritten_notes",
            "region": name.split("_")[1],
            "image": os.path.relpath(os.path.join(GNHK_DIR, name + ".jpg"), BASE),
            "reference": text,
            "allowed_extra": " ".join(printed),
            "math_tokens": math_tokens,
            "clean": clean,
            "license": "CC BY 4.0",
            "attribution": "GNHK, Lee, Chung & Lee, ICDAR 2021",
        }


def strip_markdown(md):
    md = md.replace("\\_", " ")
    # archive watermark added by the UCSF Industry Documents site, not page content
    md = re.sub(r"(?im)^.*source:\s*https?://www\.industrydocuments\.ucsf\.edu\S*.*$", " ", md)
    md = re.sub(r"~~[^~]*~~", " ", md)              # struck-through text was crossed out
    md = re.sub(r"^\s*\|?\s*[-:| ]+\|?\s*$", " ", md, flags=re.M)  # table rules
    md = re.sub(r"[|*#`>]", " ", md)
    md = re.sub(r"^\s*[-+]\s+", " ", md, flags=re.M)
    md = re.sub(r"\[[ xX]\]|[☒□]", " ", md)
    md = re.sub(r"[ \t]+", " ", md)
    return "\n".join(l.strip() for l in md.splitlines() if l.strip())


def omni_rows():
    rows = {}
    for line in open(OMNI_META):
        r = json.loads(line)
        rows[int(r["id"])] = r
    for i in OMNI_IDS:
        r = rows[i]
        md = r["true_markdown_output"]
        try:
            md = json.loads(md)
        except (ValueError, TypeError):
            pass
        yield {
            "id": f"omni_{i}",
            "source": "OmniAI",
            "category": "form_with_handwriting",
            "region": "NA",
            "image": os.path.relpath(
                os.path.join(os.path.dirname(OMNI_META), r["file_name"]), BASE
            ),
            "reference": strip_markdown(md),
            "allowed_extra": "",
            "clean": True,
            "license": "MIT",
            "attribution": "OmniAI OCR Benchmark (getomni-ai/ocr-benchmark)",
        }


def main():
    out = os.path.join(BASE, "data/manifest.jsonl")
    n = 0
    with open(out, "w") as f:
        for row in list(gnhk_rows()) + list(omni_rows()):
            assert os.path.exists(os.path.join(BASE, row["image"])), row["image"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
