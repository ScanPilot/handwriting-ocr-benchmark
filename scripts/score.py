"""Score one tool's outputs against data/manifest.jsonl.

Outputs live in runs/<tool>/<page id>.txt (plain text, any reading order).

Metrics per page, then summed over pages (micro average):
- recall: share of reference words the tool produced (multiset match, order ignored)
- precision: share of the tool's words that are in the reference; words that match
  printed text on the page (allowed_extra) are ignored rather than counted wrong
- F1 of the two
- WER / CER: ordered edit distance against the reference in line order. Secondary,
  because free-form notes have no single correct reading order.
Normalisation: Unicode NFKC, lower case, punctuation removed, hyphenated line breaks joined.

Run: python3 -I scripts/score.py <tool> [--by category|source|region] [--show N]
"""
import argparse
import collections
import json
import os
import re
import unicodedata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _number(m):
    # 2.00 == 2, .90 == 0.9, 1,250 == 1250: same value, so the same word
    s = m.group(0).replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return m.group(0)
    return ("%f" % v).rstrip("0").rstrip(".").replace(".", "d")


def words(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"-\s*\n\s*", "", text)          # na-\ntion -> nation
    text = re.sub(r"(?<![\w.])\d*[.,]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w])", _number, text)
    text = re.sub(r"[^\w\s]", " ", text)            # punctuation off, keeps digits/letters
    return text.split()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score_page(row, output):
    ref = words(row["reference"])
    hyp = words(output)
    extra = collections.Counter(words(row.get("allowed_extra", "")))
    ref_c, hyp_c = collections.Counter(ref), collections.Counter(hyp)
    hit = sum((ref_c & hyp_c).values())
    leftover = hyp_c - ref_c
    excused = sum((leftover & extra).values())
    # GNHK leaves circled list numbers (%math%) out of the reference: excuse that many
    # short numbers if the tool wrote them
    budget = row.get("math_tokens", 0)
    for w, n in sorted((leftover - extra).items()):
        if budget and w.isdigit() and len(w) <= 2:
            take = min(n, budget)
            leftover[w] -= take
            budget -= take
            excused += take
    counted_hyp = len(hyp) - excused
    # joined vs split words ("ghost busters" = "ghostbusters"), both directions
    ref_left = ref_c - hyp_c
    hyp_left = leftover - extra
    hit_ref, hit_hyp = hit, hit
    for a_seq, a_left, b_left, a_is_ref in ((ref, ref_left, hyp_left, True),
                                            (hyp, hyp_left, ref_left, False)):
        for x, y in zip(a_seq, a_seq[1:]):
            if a_left[x] > 0 and a_left[y] > 0 and b_left[x + y] > 0 and (x != y or a_left[x] > 1):
                a_left[x] -= 1
                a_left[y] -= 1
                b_left[x + y] -= 1
                if a_is_ref:
                    hit_ref, hit_hyp = hit_ref + 2, hit_hyp + 1
                else:
                    hit_ref, hit_hyp = hit_ref + 1, hit_hyp + 2
    ref_chars, hyp_chars = " ".join(ref), " ".join(hyp)
    return {
        "ref_words": len(ref),
        "hyp_words": counted_hyp,
        "hits": hit_ref,
        "hits_hyp": hit_hyp,
        "word_edits": edit_distance(ref, hyp),
        "ref_chars": len(ref_chars),
        "char_edits": edit_distance(ref_chars, hyp_chars),
        "missing": sorted(ref_left.elements())[:15],
        "invented": sorted(hyp_left.elements())[:15],
    }


def summarise(pages):
    s = collections.Counter()
    for p in pages:
        for k in ("ref_words", "hyp_words", "hits", "hits_hyp", "word_edits", "ref_chars",
                  "char_edits"):
            s[k] += p[k]
    r = s["hits"] / s["ref_words"] if s["ref_words"] else 0
    pr = min(1.0, s["hits_hyp"] / s["hyp_words"]) if s["hyp_words"] else 0
    f1 = 2 * r * pr / (r + pr) if r + pr else 0
    return {
        "pages": len(pages),
        "recall": round(r, 4),
        "precision": round(pr, 4),
        "f1": round(f1, 4),
        "wer": round(s["word_edits"] / s["ref_words"], 4) if s["ref_words"] else None,
        "cer": round(s["char_edits"] / s["ref_chars"], 4) if s["ref_chars"] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--by", default="category")
    ap.add_argument("--show", type=int, default=0, help="print the N worst pages")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    run_dir = os.path.join(BASE, "runs", a.tool)
    scored, missing = [], []
    for row in rows:
        path = os.path.join(run_dir, row["id"] + ".txt")
        if not os.path.exists(path):
            missing.append(row["id"])
            continue
        p = score_page(row, open(path, encoding="utf-8").read())
        p.update(id=row["id"], group=row.get(a.by))
        scored.append(p)
    groups = collections.defaultdict(list)
    for p in scored:
        groups[p["group"]].append(p)
    result = {"tool": a.tool, "all": summarise(scored), "missing_outputs": len(missing)}
    result["by_" + a.by] = {g: summarise(ps) for g, ps in sorted(groups.items())}
    print(json.dumps(result, indent=2))
    if a.show:
        worst = sorted(scored, key=lambda p: p["hits"] / max(p["ref_words"], 1))[: a.show]
        for p in worst:
            print(p["id"], f"recall {p['hits']}/{p['ref_words']}",
                  "missing:", p["missing"][:8], "invented:", p["invented"][:8])


if __name__ == "__main__":
    main()
