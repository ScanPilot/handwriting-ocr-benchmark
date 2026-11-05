"""Summarise repeated runs of one tool: mean and range across runs, by group.

Runs are the folders runs/<tool> and runs/<tool>-<tag> (e.g. scanpilot, scanpilot-run2).
Also counts likely spelling corrections: the reference word is not in the system
dictionary (/usr/share/dict/words), the tool wrote a dictionary word 1-2 letters away
instead. GNHK keeps writers' misspellings, so these are not reading errors.

Run: python3 -I scripts/summarise_runs.py scanpilot [--by category|region|source]
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score  # noqa: E402

BASE = score.BASE
DICT = "/usr/share/dict/words"


def load_dict():
    if not os.path.exists(DICT):
        return None
    return {w.strip().lower() for w in open(DICT)}


def near(a, b):
    return abs(len(a) - len(b)) <= 2 and score.edit_distance(list(a), list(b)) <= 2


def corrections(row, output, vocab):
    ref_c = collections.Counter(score.words(row["reference"]))
    hyp_c = collections.Counter(score.words(output))
    missing = list((ref_c - hyp_c).elements())
    invented = list((hyp_c - ref_c).elements())
    found = 0
    for m in missing:
        if len(m) < 4 or m.isdigit() or m in vocab:
            continue
        for i, w in enumerate(invented):
            if w in vocab and near(m, w):
                found += 1
                invented.pop(i)
                break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--by", default="category")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    dirs = sorted(d for d in glob.glob(os.path.join(BASE, "runs", a.tool + "*"))
                  if os.path.basename(d) == a.tool or os.path.basename(d).startswith(a.tool + "-"))
    vocab = load_dict()
    per_run = {}
    for d in dirs:
        name = os.path.basename(d)
        groups = collections.defaultdict(list)
        allp, corr, ref_total, missing_files = [], 0, 0, 0
        for row in rows:
            path = os.path.join(d, row["id"] + ".txt")
            if not os.path.exists(path):
                missing_files += 1
                continue
            out = open(path, encoding="utf-8").read()
            p = score.score_page(row, out)
            allp.append(p)
            groups[row.get(a.by)].append(p)
            if vocab and row["source"] == "GNHK":
                corr += corrections(row, out, vocab)
                ref_total += p["ref_words"]
        per_run[name] = {
            "all": score.summarise(allp),
            "groups": {g: score.summarise(ps) for g, ps in groups.items()},
            "missing_files": missing_files,
            "spelling_corrections_gnhk": corr,
            "gnhk_ref_words": ref_total,
        }

    def spread(values):
        return {"mean": round(statistics.mean(values), 4),
                "min": round(min(values), 4), "max": round(max(values), 4)}

    metrics = ("recall", "precision", "f1", "wer", "cer")
    out = {"tool": a.tool, "runs": list(per_run), "missing_files": {
        k: v["missing_files"] for k, v in per_run.items()}}
    out["all"] = {m: spread([r["all"][m] for r in per_run.values()]) for m in metrics}
    groups = sorted({g for r in per_run.values() for g in r["groups"]}, key=str)
    out["by_" + a.by] = {
        str(g): {"pages": per_run[next(iter(per_run))]["groups"].get(g, {}).get("pages"),
                 **{m: spread([r["groups"][g][m] for r in per_run.values() if g in r["groups"]])
                    for m in ("recall", "precision", "f1", "cer")}}
        for g in groups
    }
    out["spelling_corrections_gnhk"] = {
        k: f"{v['spelling_corrections_gnhk']} of {v['gnhk_ref_words']} words"
        for k, v in per_run.items()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
