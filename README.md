# DiagRare-Bench

DiagRare-Bench studies a simple question in clinical diagnosis: when the patient evidence changes, how much does a language model change its diagnostic ranking?

The evaluation separates two quantities that are usually mixed together in accuracy: a disease-prior feature and a case-specific evidence feature. For case $x$ and diagnosis $d$, we fit

$$
U_m(d,x)=\gamma_{\mathrm{prior}}^{(m)}p(d)+\gamma_{\mathrm{evidence}}^{(m)}e(d,x).
$$

The evidence coefficient is estimated from ranked differentials. We then test its interpretation on independent case reports, randomized changes to evidence strength, and sequential diagnostic revision. The released panel contains general-purpose and medically specialized open-weight checkpoints evaluated under the same protocol.

## Resources

- 🌐 Project page: https://diagrare-bench-dhariniraghavan2001-2901.vercel.app
- 📊 Leaderboard: https://diagrare-bench-dhariniraghavan2001-2901.vercel.app/leaderboard.html
- 🤗 Dataset: https://huggingface.co/datasets/Dharini24/DiagRare_Bench

## Evaluation at a glance

- 25 open-weight checkpoints
- 21 checkpoints with identifiable baseline evidence coefficients
- 984 primary diagnostic cases
- 82 diseases across nine organ systems
- 3,562 CUPCase reports and 22,901 RareArena cases
- 27 target and competing-disease pairs in the randomized evidence experiment
- 82 target and competing-disease pairs in the sequential revision task

The main cross-model associations are $r=0.843$ with top-1 accuracy, $r=0.689$ with the randomized evidence effect, and $r=0.860$ with sequential recovery. The paper reports the exact model count and inclusion rule for each analysis.

## Repository structure

- `code/benchmark_generation/` builds the primary diagnostic cases.
- `code/inference/` contains open-weight inference and prompt definitions.
- `code/analysis/` contains Plackett-Luce estimation, scoring, and figure generation.
- `code/theory/` contains the randomized, sequential, external-transfer, and robustness analyses.
- `results/` contains the machine-readable tables used in the paper.
- `figures/` contains the publication figures in SVG.
- `paper/` contains anonymous and public manuscript sources.
- `site/` contains the project page and leaderboard.

## Reproducing the main coefficient

The central estimator is implemented in `code/analysis/plackett_luce.py`. The baseline model table is rebuilt with `code/analysis/fit_per_all_models.py` once the corresponding prediction files are available. The remaining analyses are grouped by scientific question rather than by compute job.

## External datasets

CUPCase and RareArena remain separate external datasets and retain their original licenses and attribution. This repository contains analysis code and derived result tables, not a new license for those resources.

## Citation

```bibtex
@article{diagrarebench2026,
  title   = {DiagRare-Bench: Separating Disease Priors from Patient Evidence in Language Models for Clinical Diagnosis},
  author  = {TBD},
  journal = {Preprint},
  year    = {2026}
}
```
