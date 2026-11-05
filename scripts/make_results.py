"""Score every system from runs/ and write results/results.json and results/results.md.

The reported setting of each system is fixed here, so the published table can be rebuilt
exactly. Systems with several runs (ScanPilot, 3 runs) are reported as the mean, with the
range in results.json.

Run (from the repo root): python scripts/make_results.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score  # noqa: E402

BASE = score.BASE

# name shown in tables -> run folders (the reported setting)
SYSTEMS = {
    "ScanPilot": ["scanpilot", "scanpilot-run2", "scanpilot-run3"],
    "Azure Document Intelligence (Read)": ["azure"],
    "Google Cloud Vision (handwriting hint)": ["vision-hw"],
    "Amazon Textract": ["textract"],
    "TrOCR large handwritten (PaddleOCR detector)": ["trocr"],
    "PaddleOCR 3.4.1 (PP-OCRv5)": ["paddleocr"],
    "Tesseract 5.5.1": ["tesseract"],
    "EasyOCR 1.7.2": ["easyocr"],
}
# which group each system belongs to (shown in results.json and results.md)
KIND = {
    "ScanPilot": "AI model",
    "Azure Document Intelligence (Read)": "Cloud OCR",
    "Google Cloud Vision (handwriting hint)": "Cloud OCR",
    "Amazon Textract": "Cloud OCR",
    "TrOCR large handwritten (PaddleOCR detector)": "Open-source OCR",
    "PaddleOCR 3.4.1 (PP-OCRv5)": "Open-source OCR",
    "Tesseract 5.5.1": "Open-source OCR",
    "EasyOCR 1.7.2": "Open-source OCR",
}
# other settings we ran, reported in results.json only
EXTRA = {
    "Google Cloud Vision (no hint)": ["vision"],
    "Tesseract 5.5.1 (--psm 11)": ["tesseract-psm11"],
}
GROUPS = {
    "All pages": lambda r: True,
    "Handwritten notes (GNHK)": lambda r: r["category"] == "handwritten_notes",
    "Forms with handwriting (OmniAI)": lambda r: r["category"] == "form_with_handwriting",
    "Clean pages": lambda r: r["clean"],
}
for region in ("AF", "AS", "EU", "NA"):
    GROUPS[f"Region {region}"] = (lambda reg: lambda r: r["region"] == reg)(region)
METRICS = ("recall", "precision", "f1", "wer", "cer")


def score_run(rows, folder):
    pages = {}
    for row in rows:
        path = os.path.join(BASE, "runs", folder, row["id"] + ".txt")
        out = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        pages[row["id"]] = score.score_page(row, out)
    return pages


def system_result(rows, folders):
    runs = [score_run(rows, f) for f in folders]
    res = {"runs": folders, "groups": {}}
    for gname, pick in GROUPS.items():
        ids = [r["id"] for r in rows if pick(r)]
        per_run = [score.summarise([run[i] for i in ids]) for run in runs]
        res["groups"][gname] = {"pages": len(ids), **{
            m: {"mean": round(statistics.mean(p[m] for p in per_run), 4),
                "min": round(min(p[m] for p in per_run), 4),
                "max": round(max(p[m] for p in per_run), 4)} for m in METRICS}}
    res["missed_words_notes"] = round(1 - res["groups"]["Handwritten notes (GNHK)"]["recall"]["mean"], 4)
    res["pages_scored"] = len(rows)
    res["empty_outputs"] = {f: sum(1 for i, p in run.items() if p["hyp_words"] == 0)
                            for f, run in zip(folders, runs)}
    return res


def main():
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data/manifest.jsonl"))]
    out = {"pages": len(rows), "metric_notes": "word recall/precision/F1, order ignored; "
           "WER/CER ordered, secondary (see README)",
           "systems": {n: {"group": KIND[n], **system_result(rows, f)} for n, f in SYSTEMS.items()},
           "other_settings": {n: system_result(rows, f) for n, f in EXTRA.items()}}
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(BASE, "results/results.json"), "w"), indent=2)

    names = list(SYSTEMS)
    lines = ["# Results: word F1 (recall / precision)", "",
             "| Pages | Count | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names),
             "| Group | | " + " | ".join(KIND[n] for n in names) + " |"]
    for g in GROUPS:
        cells = []
        for n in names:
            d = out["systems"][n]["groups"][g]
            cells.append(f"**{d['f1']['mean']*100:.1f}%** ({d['recall']['mean']*100:.1f} / "
                         f"{d['precision']['mean']*100:.1f})")
        lines.append(f"| {g} | {out['systems'][names[0]]['groups'][g]['pages']} | " + " | ".join(cells) + " |")
    lines += ["", "Missed words on handwritten notes: " + ", ".join(
        f"{n} {out['systems'][n]['missed_words_notes']*100:.1f}%" for n in names), "",
        "ScanPilot: mean of 3 runs; per-run range in results.json. Region NA includes the 11 forms."]
    open(os.path.join(BASE, "results/results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
