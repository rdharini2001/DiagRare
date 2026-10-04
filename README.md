# DiagRare-Bench

**Separating Disease Priors from Patient Evidence in Language Models for Clinical Diagnosis**

Dharini Raghavan · Amritpal Singh

[Paper](https://openreview.net/forum?id=0Eua4EkYht) · [Dataset](https://huggingface.co/datasets/Dharini24/DiagRare_Bench) · [Project page](https://diagrare-bench-dhariniraghavan2001-2901.vercel.app) · [Leaderboard](https://diagrare-bench-dhariniraghavan2001-2901.vercel.app/leaderboard.html)

DiagRare-Bench measures how strongly a model's diagnostic ranking changes with patient evidence after a disease-prior feature is accounted for. The evaluation is designed to complement diagnostic accuracy: two models can reach similar accuracy while responding differently when the evidence itself changes.

## What is released

```text
.
├── code/           # benchmark construction, inference, estimation, robustness, interventions
├── data/           # benchmark-owned structured, intervention, metadata, and adaptation data
├── tables/         # machine-readable result tables used in the paper
├── figures/        # publication SVGs
├── leaderboard/    # static interactive model table + leaderboard.csv
├── site/           # project page, deployable as a static site
├── paper/          # public preprint PDF
├── requirements.txt
├── CITATION.cff
├── NOTICE.md
└── LICENSE
```

## Benchmark at a glance

- 984 primary diagnostic cases covering 82 diseases and nine organ systems.
- 25 open-weight checkpoints spanning general-purpose and medically specialized models.
- A randomized 3 × 3 evidence/stated-prevalence experiment over 27 disease pairs.
- A sequential diagnostic-revision task over 82 disease pairs.
- Transfer analyses on CUPCase and RareArena using a different evidence representation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU inference scripts use `vllm`, `torch`, and `transformers`; analysis-only workflows can be run with the NumPy/pandas/SciPy/Matplotlib/Statsmodels subset of the requirements.

## Core analysis

The primary model is

```text
U_m(d, x) = gamma_prior * p(d) + gamma_evidence * e(d, x)
```

where `p(d)` is the released disease-prior feature and `e(d, x)` is case-specific evidence. The fitted `gamma_evidence` is interpreted comparatively under a fixed evaluation protocol.

Key scripts:

- `code/analysis/plackett_luce.py` — Plackett-Luce estimator.
- `code/analysis/fit_per_all_models.py` — primary coefficient estimation.
- `code/analysis/score_predictions.py` — diagnostic accuracy and ranking statistics.
- `code/theory/analyze_causal_grid.py` — randomized intervention analysis.
- `code/theory/analyze_sequential_anchoring.py` — sequential recovery and order dependence.
- `code/analysis/make_publication_figures.py` — publication figures from released result tables.

## Reproduce figures

From the repository root:

```bash
python code/analysis/make_publication_figures.py
```

Generated figures are written to `figures/`.

## Data

The benchmark-owned data are also published as a Hugging Face dataset: `https://huggingface.co/datasets/Dharini24/DiagRare_Bench`. The repository copy under `data/` is included so the analysis scripts can run from a single checkout.

CUPCase and RareArena are external datasets and are not redistributed. The repository includes preparation and analysis code for those evaluations; obtain the source datasets from their original repositories and follow their original licenses.

## Leaderboard

`leaderboard/leaderboard.csv` is generated from the released result tables. `leaderboard/index.html` is a static interactive view with search, filtering, sorting, upstream model links, and no imputation of missing analysis-specific estimates.

## Scope

DiagRare-Bench is a research benchmark for diagnostic model behavior. It does not evaluate clinical safety or establish that any checkpoint is suitable for patient care. Evidence responsiveness is protocol-dependent and should be reported alongside conventional diagnostic performance rather than treated as a deployment score.

## Citation

```bibtex
@misc{raghavan2026diagrarebench,
  title  = {{DiagRare-Bench: Separating Disease Priors from Patient Evidence in Language Models for Clinical Diagnosis}},
  author = {Raghavan, Dharini and Singh, Amritpal},
  year   = {2026},
  note   = {ICLR 2027 Conference Submission},
  url    = {https://openreview.net/forum?id=0Eua4EkYht}
}
```
