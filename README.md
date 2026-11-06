# Handwriting OCR benchmark

**Results, charts and findings: [Handwriting OCR benchmark on scanpilot.ai](https://www.scanpilot.ai/research/handwriting-ocr-benchmark/)**

How well do free OCR tools, cloud OCR services and AI models read handwriting? We ran eight
systems on the same 183 public pages (172 phone photos of handwritten notes and 11 scanned forms with handwritten
fill-ins) and scored every page the same way. This repo holds everything needed to check or
repeat the test: the answer keys, every system's output for every page, the scorer, the
runners and the exact settings.

Published by SUSTRAVIA GmbH, the company behind ScanPilot, which is one of the systems tested.
See [Disclosure](#disclosure) before you read the numbers.

## Results (English, v1)

Word F1: the balance of finding the words on the page (recall) and not writing wrong ones
(precision). Higher is better.

| System | Group | All pages (183) | Handwritten notes (172) | Forms with handwriting (11) | Clean pages (101) | Words missed on notes |
|---|---|---|---|---|---|---|
| ScanPilot | AI model | **91.6%** | **90.9%** | 94.6% | **94.5%** | 7.1% |
| Azure Document Intelligence (Read) | Cloud OCR | 88.0% | 86.6% | 94.5% | 92.9% | 11.3% |
| Google Cloud Vision (handwriting hint) | Cloud OCR | 86.2% | 84.5% | 94.1% | 91.9% | 13.4% |
| Amazon Textract | Cloud OCR | 86.0% | 84.1% | 94.7% | 90.8% | 13.5% |
| TrOCR large handwritten (PaddleOCR detector) | Open-source OCR | 67.0% | 64.3% | 79.5% | 72.1% | 34.5% |
| PaddleOCR 3.4.1 (PP-OCRv5) | Open-source OCR | 54.4% | 48.9% | 79.2% | 65.1% | 53.1% |
| Tesseract 5.5.1 | Open-source OCR | 24.4% | 13.1% | 68.3% | 36.0% | 89.5% |
| EasyOCR 1.7.2 | Open-source OCR | 23.1% | 15.0% | 61.0% | 33.0% | 84.8% |

- **General open-source OCR does not read handwritten notes.** Tesseract finds 10.5% of the
  handwritten words and EasyOCR 15.2%. PaddleOCR, whose recogniser is documented as supporting
  handwriting, finds about half. TrOCR, a model trained on handwritten lines, is the best free
  option at 64.3% F1.
- **Forms are close to solved, notes are not.** The three cloud services are within 0.6 points of
  each other on the forms (94.1 to 94.7%) and drop to 84 to 87% on free handwritten notes, where
  they miss 11 to 14% of the words.
- **ScanPilot misses the fewest words on notes** (7.1%, against 11.3% for the best cloud OCR).
  ScanPilot is our product and the only AI model in this version; see [Disclosure](#disclosure).
- The three groups rank the same way in every writer region in GNHK (Africa, Asia, Europe,
  North America): ScanPilot first, then the cloud services, then the open-source tools. Within
  a group some places swap by region; see [results/results.md](results/results.md). With 11
  forms there is no winner on that row.
- Full numbers with recall and precision, per-region results, ScanPilot's run-to-run range and
  other settings we tried: [results/results.json](results/results.json).
- Speed is not compared. The open-source tools ran on one laptop, where memory pressure made
  some pages take minutes, and the cloud times depend on network and region.

## What was tested

| System | What we called | Settings | Date run |
|---|---|---|---|
| ScanPilot | ScanPilot's converter (layout mode), the same code path as the free demo on [scanpilot.ai](https://www.scanpilot.ai/); backend commit d37fda7 | Page sent as a one-page PDF (150 dpi); the returned grid flattened row by row | 2026-11-03, 3 runs |
| Azure Document Intelligence | `prebuilt-read`, API 2024-11-30 (v4.0 GA), Standard S0, region Germany West Central | Defaults (Read has no handwriting or language setting) | 2026-11-05, 1 run |
| Google Cloud Vision | `images:annotate` with `DOCUMENT_TEXT_DETECTION` | `languageHints: ["en-t-i0-handwrit"]`, Google's handwriting hint (without it: 85.0% all pages) | 2026-11-03, 1 run |
| Amazon Textract | `DetectDocumentText`, Region eu-central-1 | Defaults (no handwriting setting) | 2026-11-03, 1 run |
| TrOCR | [microsoft/trocr-large-handwritten](https://huggingface.co/microsoft/trocr-large-handwritten), transformers 5.22.1 | TrOCR reads one line at a time and has no detector, so lines come from PaddleOCR's PP-OCRv5_mobile_det (polygons in `runs/_detect/`); each line's box cropped with 4 px padding, read in batches of 8, lines ordered top to bottom, left to right | 2026-11-05, 1 run |
| PaddleOCR 3.4.1 | PP-OCRv5 pipeline: PP-OCRv5_mobile_det, PP-OCRv5_server_rec, PP-LCNet_x1_0_textline_ori; PaddlePaddle 3.3.1, CPU | Detector input capped at 2,560 px; page-orientation and unwarping models off (pages are already upright). The default server detector needed 21 to 47 GB of memory on 4,000 px photos; see `scripts/run_paddleocr.py` | 2026-11-05, 1 run |
| Tesseract 5.5.1 | `tesseract -l eng --oem 1 --psm 3` (LSTM, automatic layout) | `--psm 11` sparse-text mode scored lower (19.7%), kept as a check | 2026-11-03, 1 run |
| EasyOCR 1.7.2 | `easyocr.Reader(["en"])`, `readtext(detail=0)`, PyTorch 2.15 | Defaults (CRAFT detector, canvas 2,560 px) | 2026-11-05, 1 run |

Every system got the same upright page. 37 GNHK photos store their rotation only as an EXIF
flag; every runner applies it first (PIL `exif_transpose`). The three cloud services received
identical JPEG bytes (quality 95, full resolution); the open-source tools got the same upright
image in memory. Cloud OCR and the open-source tools are deterministic, so one run each; ScanPilot is not, so we ran it three times and report the mean (F1 on notes
ranged 90.7 to 91.1%).

## Data

| Source | Pages | What | Licence |
|---|---|---|---|
| [GNHK](https://github.com/GoodNotes/GNHK-dataset) test split | 172 | Phone photos of handwritten notes, writers from Africa (29), Asia (25), Europe (82) and North America (36); 9,199 handwritten words | CC BY 4.0, Lee, Chung & Lee, ICDAR 2021 |
| [OmniAI OCR benchmark](https://huggingface.co/datasets/getomni-ai/ocr-benchmark) | 11 | Scanned forms with handwritten fill-ins: ids 880, 661, 46, 36, 40, 54, 426, 43, 26, 430, 41 | MIT |

- The answer keys come from the dataset authors' own annotations, not from any system tested.
- The OmniAI dataset has no handwriting tag, so we picked the 11 forms by eye from its 124 real
  scans (the rest of its 1,000 pages are rendered in fonts). We picked them before running any
  system, and the list has not changed.
- GoodNotes' original GNHK download page and Google Drive folders are no longer online. We use
  the copy on Hugging Face (staghado/GNHK-Dataset), the same size as the original download;
  `data/checksums.sha256` lists the exact images scored.
- The images are not stored in this repo. `scripts/download_data.py` fetches them from the
  sources above and checks them against the checksums.

## Scoring (scripts/score.py)

- **Primary: word recall, precision and F1, order ignored.** Free-form notes have no single
  correct reading order, and two-column pages read row by row would be punished for nothing.
  Ordered word and character error rates (WER, CER) are in results.json as a secondary measure.
- Normalisation: Unicode NFKC, lower case, punctuation removed, words split across a line with a
  hyphen joined. Numbers compare by value (2.00 = 2). Joined or split words count as the same
  word ("ghost busters" = "ghostbusters").
- GNHK: printed words on the page are not required, and writing them is not counted wrong.
  Unreadable, crossed-out and math tokens are left out of the answer key; circled list numbers
  the key leaves out are not counted wrong (up to the number of math tokens on the page).
- **Clean pages:** GNHK pages with nothing crossed out, unreadable or mathematical, plus the 11
  forms. On the other pages the answer key leaves out text that is visibly on the page, so every
  system's precision is understated there. Use the clean group for precision.
- A system that returns nothing for a page scores zero on that page.

## Limitations

- **English only.** Sections for the other languages ScanPilot serves (Hindi, Bengali, German,
  Japanese, Chinese and more) will be added as new pages with double-checked transcriptions.
- **GNHK keeps the writers' misspellings** ("memtorship"). A system that corrects spelling loses
  points. ScanPilot corrected about 100 of 8,635 GNHK words per run (1.2%), and those count as
  misses here.
- **11 forms is a small sample.** Treat the forms row as "all close", not as a ranking.
- **One page fails for ScanPilot:** gnhk_eng_EU_216, a student's handwritten copy of a published
  text. ScanPilot's model declines to transcribe it and ScanPilot returns an error, so the page
  scores zero for ScanPilot in all three runs (runs/failures.json).
- **PaddleOCR's recogniser is multilingual** and wrote 75 of its 9,342 words (0.8%) in Chinese,
  Japanese or Korean characters. They count as wrong words, as they would for a user.
- **Services change.** These are the results on the dates above. We will rerun when the
  services or models update and log every change in [CHANGELOG.md](CHANGELOG.md).
- **One AI model.** ScanPilot is the only AI system in this version. General AI models on their
  own (such as ChatGPT, Claude or open vision-language models) and phone apps (Google Lens,
  Microsoft OneNote, Adobe Scan) were not tested, so this repo says nothing about them.
- **Open-source tools were run with their documented defaults** on an Apple laptop (CPU for
  PaddleOCR, Apple MPS for EasyOCR and TrOCR). Fine-tuning any of them on handwriting would
  change their numbers; this measures what they do out of the box.

## Disclosure

- ScanPilot is our product: an AI model with our own page handling and layout step around it.
- We ran every system ourselves: the cloud services on normal paid accounts, the open-source
  tools on our own machine, all with default settings plus each provider's documented
  handwriting option where one exists.
- ScanPilot's code is not open, so its runner is not in this repo. Its outputs for every page
  are, and anyone can run the same pages through ScanPilot's public converter.
- If you think a system was set up unfairly, open an issue with the setting you would use. We
  will rerun it and publish the result, including if it beats ScanPilot.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py        # fetches the 183 images and verifies the checksums
python scripts/make_results.py         # rescores every system from runs/ -> results/
```

To rerun a system yourself (outputs go to `runs/<system>-<tag>/`, so the published runs stay
untouched):

```bash
export GOOGLE_VISION_API_KEY=...       # Google Cloud API key with the Vision API enabled
python scripts/run_vision.py --hint handwriting --tag mine

export AZURE_DI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
export AZURE_DI_KEY=...
python scripts/run_azure.py --tag mine

python scripts/run_textract.py --tag mine          # standard AWS credentials
python scripts/run_tesseract.py --tag mine

pip install -r requirements-open-source.txt        # EasyOCR + TrOCR environment
python scripts/run_easyocr.py --tag mine
python scripts/run_trocr.py --tag mine             # uses the published line polygons

# PaddleOCR in its own environment: pip install paddlepaddle==3.3.1 paddleocr==3.4.1 pillow
python scripts/run_paddleocr.py --tag mine
python scripts/run_paddle_detect.py                # only to rebuild the TrOCR line polygons

python scripts/score.py vision-mine                # score any run folder
```

`scripts/build_manifest.py` rebuilds `data/manifest.jsonl` (the answer keys) from the downloaded
annotations, if you want to check that step as well.

## Repo layout

| Path | What |
|---|---|
| `data/manifest.jsonl` | One row per page: id, source, category, region, image path, answer key, printed words allowed, licence |
| `data/checksums.sha256` | SHA-256 of every image scored |
| `runs/<system>/<page id>.txt` | Each system's raw text output for each page |
| `runs/_detect/paddle_mobile/<page id>.json` | Line polygons from PaddleOCR's detector, the input to TrOCR |
| `results/` | Scores generated by `scripts/make_results.py` |
| `scripts/` | Runners, scorer, data download |

## Licence

- Code (`scripts/`): MIT, see [LICENSE](LICENSE).
- Answer keys in `data/manifest.jsonl`: derived from GNHK (CC BY 4.0) and the OmniAI OCR
  benchmark (MIT); keep their attribution, see [DATA_LICENSE.md](DATA_LICENSE.md).
- System outputs (`runs/`) and results (`results/`): CC BY 4.0, credit "ScanPilot handwriting
  OCR benchmark".
- GNHK's authors remove pages on request; if a page is removed upstream, we remove it here.

## Cite

See [CITATION.cff](CITATION.cff). Questions, reruns, removal requests: hello@sustravia.com.
