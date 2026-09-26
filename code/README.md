# DiagRare-Bench code

This directory contains the scripts used to construct the benchmark, run open-weight model inference, estimate evidence responsiveness, analyze the intervention experiments, and reproduce the reported robustness checks.

All evaluations in this submission use open-weight checkpoints. The same analysis code is used for general-purpose and medically specialized models.

## Main analysis

- `analysis/plackett_luce.py`: Plackett-Luce estimator for ranked diagnostic outputs.
- `analysis/fit_per_all_models.py`: fits the prior and evidence coefficients for the model panel.
- `analysis/score_predictions.py`: computes diagnostic accuracy and ranking statistics.
- `analysis/make_publication_figures.py`: reproduces the publication figures from included result tables.

## Benchmark construction

- `benchmark_generation/generate_vignettes.py`: generates the primary diagnostic cases from the disease ontology.
- `benchmark_generation/new_diseases.py`: definitions used to expand the disease set.

## Inference

- `inference/run_inference_vllm.py`: open-weight inference with vLLM.
- `inference/run_model_all_conditions.py`: common runner for the primary benchmark and intervention conditions.
- `inference/prompts.py`: prompt templates and output-format instructions.

## Evidence interventions

- `theory/causal_intervention_grid.py`: builds the 3 by 3 evidence and stated-prior experiment.
- `theory/analyze_causal_grid.py`: estimates intervention effects.
- `theory/causal_response_surface.py`: response-surface and cross-model validation analyses.
- `theory/build_sequential_anchoring_cases.py`: constructs the sequential diagnostic-revision task.
- `theory/analyze_sequential_anchoring.py`: recovery and order-dependence analyses.

## External case reports

The scripts under `theory/external_validation/` prepare, score, and analyze CUPCase and RareArena. The external datasets retain their original licenses and are not relicensed by this repository.

## Robustness

The scripts under `theory/robustness/` cover alternative choice likelihoods, candidate-set dependence, clustered uncertainty, feature-shuffling tests, matched rare/common analyses, family-adjusted regressions, and related sensitivity checks.

## Fine-tuning and inference-time analyses

- `finetuning/`: QLoRA data preparation and adaptation scripts.
- `analysis/self_consistency_analysis.py` and `theory/robustness/self_consistency_calibration.py`: self-consistency analyses.
- `theory/analyze_qwen3_thinking_budget.py`: Qwen3 thinking-budget analysis.

## Reproducing the figures

From the repository root:

```bash
python code/analysis/make_publication_figures.py
```

The script reads the included result tables and writes vector figures to `figures/`.
