"""Download the 183 test images from their original public sources and verify them.

- GNHK test split (172 photos): Hugging Face dataset staghado/GNHK-Dataset, file GNHK-dataset.zip
  (1.0 GB; the GoodNotes download page and Google Drive folders are gone). Only test_data.zip is
  extracted, to data/gnhk-raw/test/.
- OmniAI ocr-benchmark (11 forms): Hugging Face dataset getomni-ai/ocr-benchmark, test split,
  only the 11 pages used here, to data/omni-raw/test/.

Every image is then checked against data/checksums.sha256, so you score exactly the files we did.
data/manifest.jsonl (with the reference text) is already in the repo; scripts/build_manifest.py
rebuilds it from these downloads if you want to check that step too.

Run (from the repo root, needs huggingface_hub):
  python scripts/download_data.py
"""
import hashlib
import json
import os
import zipfile

from huggingface_hub import hf_hub_download

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OMNI_IDS = [880, 661, 46, 36, 40, 54, 426, 43, 26, 430, 41]


def gnhk():
    target = os.path.join(DATA, "gnhk-raw")
    if os.path.isdir(os.path.join(target, "test", "test")):
        print("GNHK already extracted")
        return
    zpath = hf_hub_download("staghado/GNHK-Dataset", "GNHK-dataset.zip", repo_type="dataset",
                            local_dir=target)
    with zipfile.ZipFile(zpath) as outer:
        outer.extract("test_data.zip", target)
    with zipfile.ZipFile(os.path.join(target, "test_data.zip")) as inner:
        inner.extractall(os.path.join(target, "test"))
    print("GNHK test split extracted")


def omni():
    target = os.path.join(DATA, "omni-raw")
    meta = hf_hub_download("getomni-ai/ocr-benchmark", "test/metadata.jsonl", repo_type="dataset",
                           local_dir=target)
    rows = {int(json.loads(l)["id"]): json.loads(l) for l in open(meta)}
    for i in OMNI_IDS:
        hf_hub_download("getomni-ai/ocr-benchmark", "test/" + rows[i]["file_name"],
                        repo_type="dataset", local_dir=target)
    print("OmniAI pages downloaded:", len(OMNI_IDS))


def verify():
    bad = 0
    for line in open(os.path.join(DATA, "checksums.sha256")):
        digest, rel = line.split(maxsplit=1)
        path = os.path.join(BASE, rel.strip())
        if not os.path.exists(path):
            print("missing", rel.strip())
            bad += 1
        elif hashlib.sha256(open(path, "rb").read()).hexdigest() != digest:
            print("checksum differs", rel.strip())
            bad += 1
    print("all 183 images verified" if not bad else f"{bad} problem(s)")
    return bad == 0


if __name__ == "__main__":
    gnhk()
    omni()
    raise SystemExit(0 if verify() else 1)
