---
title: DiagRare-Bench Leaderboard
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: true
short_description: Do LLM diagnoses follow evidence or prevalence?
datasets:
- Dharini24/DiagRare_Bench
tags:
- leaderboard
- benchmark
- medical
- clinical-diagnosis
- diagnostic-reasoning
- evidence-responsiveness
- plackett-luce
- llm-evaluation
- eval:ranking
- test:public
- language:english
- domain:medical
---

# DiagRare-Bench Leaderboard

Interactive leaderboard for **DiagRare-Bench**, a benchmark of how strongly language models' ranked differential diagnoses follow patient evidence once disease prevalence is held fixed. It reports 32 open-weight checkpoints (0.5B to 72B) on 984 structured cases covering 82 diseases and nine organ systems.

**[Paper](https://openreview.net/forum?id=0Eua4EkYht) · [Dataset](https://huggingface.co/datasets/Dharini24/DiagRare_Bench) · [Code](https://github.com/rdharini2001/DiagRare)**

The page is self-contained: sorting, filtering, the comparison explorer, intervention and transfer plots, and the referral-context view all run in the browser. No backend or token is needed.

## Files

| File | Purpose |
|---|---|
| `index.html`, `styles.css`, `app.js` | the page |
| `results.json`, `results.js` | all numbers shown on the page (`results.js` is the same data for offline use) |
| `leaderboard.csv` | flat table of the same results |
| `build_results.py` | regenerates the three data files from the released tables in the code repository |

## How checkpoints are ranked

A checkpoint is ranked by its evidence coefficient γ_evidence only if the coefficient is identifiable, at least half of its outputs are usable, and its top-1 accuracy is at least 5%. Other checkpoints are shown without a rank. No value is imputed: a measure that was not run for a checkpoint is shown as a dash. Seven larger checkpoints added during review have baseline results only.

γ_evidence depends on the evidence scale, prompt, candidate set and decoding rule, and is reported next to accuracy. It is not a clinical safety score.

## Local preview

Open `index.html` in a browser, or serve this folder with `python3 -m http.server 8000`.
