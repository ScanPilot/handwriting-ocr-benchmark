# Data licences and attribution

## Answer keys (data/manifest.jsonl)

Derived from two public datasets. If you reuse them, keep both attributions.

- **GNHK** (172 pages): Alex W. C. Lee, Jonathan Chung, Marco Lee. "GNHK: A Dataset for English
  Handwriting in the Wild". ICDAR 2021. Licence: CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/GoodNotes/GNHK-dataset
  (copy used: https://huggingface.co/datasets/staghado/GNHK-Dataset).
  Changes we made: kept handwritten words only, in line order; removed unreadable, crossed-out
  and math tokens; printed words listed separately as "allowed_extra".
- **OmniAI OCR benchmark** (11 pages): getomni-ai/ocr-benchmark, MIT licence, Copyright (c) OmniAI
  (https://huggingface.co/datasets/getomni-ai/ocr-benchmark).
  Changes we made: reference markdown converted to plain text (markdown syntax and a source
  watermark line removed).

## Images

Not redistributed here. `scripts/download_data.py` fetches them from the sources above, under
the same licences. GNHK's authors remove pages on request; we follow upstream removals.

## System outputs and results (runs/, results/)

CC BY 4.0, credit: "ScanPilot handwriting OCR benchmark (SUSTRAVIA GmbH)". The outputs are the
raw text each service returned for the public pages above.
