<p align="center">
  <img src="document/figures/fig3_randomized_response.png" alt="DiagRare-Bench randomized evidence and stated-prior experiment" width="100%">
</p>

<a href="https://openreview.net/forum?id=0Eua4EkYht"><img src="https://img.shields.io/badge/Paper-OpenReview-8A2BE2"></a>
<a href="https://huggingface.co/datasets/Dharini24/DiagRare_Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow"></a>
<a href="https://diagrare-bench-dhariniraghavan2001-2901.vercel.app"><img src="https://img.shields.io/badge/Project-Page-0E7C6B"></a>
<a href="https://diagrare-bench-dhariniraghavan2001-2901.vercel.app/leaderboard.html"><img src="https://img.shields.io/badge/Interactive-Leaderboard-2C6CB0"></a>

# DiagRare-Bench

**DiagRare-Bench** studies a specific question in clinical diagnosis: **when patient evidence changes, how much does a language model change its diagnostic ranking?**

The benchmark separates a disease-prior feature from a case-specific evidence feature in ranked diagnostic outputs using a two-feature Plackett-Luce model. The fitted evidence coefficient, $\gamma_{\mathrm{evidence}}$, measures how strongly fitted pairwise diagnostic log odds change with an evidence contrast when the prior contrast is held fixed. The coefficient is then tested against independent case-report datasets, randomized evidence changes, and sequential diagnostic revision.

> [!NOTE]
> Evidence responsiveness is an output-level, protocol-relative description of diagnostic decision behavior. It is **not** a clinical safety score and is not a claim about the model's internal causal mechanism.

# Paper

**DiagRare-Bench: Separating Disease Priors from Patient Evidence in Language Models for Clinical Diagnosis**  
Dharini Raghavan, Amritpal Singh  
*ICLR 2027 Conference Submission*  
<a href="https://openreview.net/forum?id=0Eua4EkYht"><img src="https://img.shields.io/badge/Paper-OpenReview-purple"></a>
<a href="document/DiagRare_Bench_ICLR2027.pdf"><img src="https://img.shields.io/badge/PDF-Repository-orange"></a>

# Benchmark Components

- **Primary diagnostic evaluation:** 984 cases spanning **82 diseases** and **nine organ systems**. Each case contains positive findings and explicitly absent findings; rare and common targets are balanced so that prevalence alone cannot solve the task.
- **Randomized evidence and stated-prior experiment:** a **3 x 3 factorial design** over 27 target/confounder pairs, with evidence strength and stated prevalence manipulated independently.
- **Sequential diagnostic revision:** 82 target/confounder pairs presented in opposite evidence orders to measure recovery after an initial competing diagnosis and final-step order dependence.
- **Negative control and robustness suite:** irrelevant-text perturbations, alternative choice likelihoods, prevalence representations, candidate sets, evidence-score parameterizations, clustered uncertainty, and model-family-adjusted analyses.
- **Model panel:** 25 open-weight checkpoints, including both general-purpose and medically specialized models, evaluated under the same diagnostic protocol.

<p align="center">
  <img src="document/figures/fig1_accuracy.png" alt="Evidence responsiveness and top-1 diagnostic accuracy" width="95%">
</p>

# Main Findings

1. **Diagnostic accuracy:** across the 21 checkpoints with identifiable baseline estimates, evidence responsiveness is associated with top-1 accuracy ($r=0.843$). This same-case association is descriptive, not the primary validation.
2. **External transfer:** the cross-model ordering transfers to all 3,562 CUPCase reports ($r=0.662$) and all 22,901 RareArena cases ($r=0.790$) under a different evidence representation.
3. **Randomized evidence:** baseline $\gamma_{\mathrm{evidence}}$ predicts the finite-difference effect of stronger assigned evidence ($r=0.689$, $p=5.5\times10^{-4}$). The related ordinal intervention coefficient gives $r=0.826$ ($p=4.0\times10^{-6}$).
4. **Sequential revision:** baseline $\gamma_{\mathrm{evidence}}$ predicts recovery after an initial competing diagnosis ($r=0.860$, $p=9.8\times10^{-6}$, $n=17$).
5. **Prior-side asymmetry:** the prior coefficient is retained as a conditioning coordinate; it is not interpreted as a separately validated index of prevalence bias.

<p align="center">
  <img src="document/figures/fig2_randomized_validation.png" alt="Randomized evidence validates the rank-derived coefficient" width="92%">
</p>

<p align="center">
  <img src="document/figures/fig4_sequential_revision.png" alt="Evidence responsiveness predicts sequential diagnostic revision" width="100%">
</p>

# Download Data

The benchmark-owned dataset is hosted on Hugging Face:

- **Dataset:** https://huggingface.co/datasets/Dharini24/DiagRare_Bench
- **Code:** https://github.com/rdharini2001/DiagRare
- **Project page:** https://diagrare-bench-dhariniraghavan2001-2901.vercel.app
- **Leaderboard:** https://diagrare-bench-dhariniraghavan2001-2901.vercel.app/leaderboard.html

```bash
pip install -U "huggingface_hub>=0.34"
hf download Dharini24/DiagRare_Bench --repo-type dataset --local-dir ./DiagRare_Bench
```

# Repository Structure

```text
DiagRare/
├── code/                 # benchmark construction, inference, estimation, interventions, robustness
├── data/                 # benchmark-owned structured/intervention/metadata/fine-tuning data
├── tables/               # machine-readable analysis tables used in the paper
├── document/
│   ├── DiagRare_Bench_ICLR2027.pdf
│   ├── supplementary.pdf
│   └── figures/          # main-paper and supplementary figures
├── site/                 # static project page + interactive leaderboard
├── requirements.txt
├── CITATION.cff
├── NOTICE.md
└── LICENSE
```

# Reproducing the Publication Figures

From the repository root:

```bash
python code/analysis/make_publication_figures.py
```

The script reads the included result tables and writes generated figures to `figures/`. The package also includes the exact main-paper figures under `document/figures/` for reference.

# External Case Reports

The paper evaluates all **3,562 CUPCase** reports and all **22,901 RareArena** cases. These datasets retain their original licenses and attribution and are not redistributed in this repository. The release contains the preparation/analysis code and the paper reports transfer through cross-model correlations because the external evidence representation differs from the primary structured representation.

# Intended Use and Limitations

DiagRare-Bench is intended for research on language-model behavior in clinical diagnosis. The primary cases use a fixed disease vocabulary and simplified disease-finding relations so that prior and evidence can be represented separately. External lexical evidence features are proxies rather than calibrated clinical likelihoods. The benchmark does not establish that any model is safe or reliable for patient care.

# Citation

```bibtex
@misc{raghavan2026diagrarebench,
  title  = {{DiagRare-Bench: Separating Disease Priors from Patient Evidence in Language Models for Clinical Diagnosis}},
  author = {Raghavan, Dharini and Singh, Amritpal},
  year   = {2026},
  note   = {ICLR 2027 Conference Submission},
  url    = {https://openreview.net/forum?id=0Eua4EkYht}
}
```
