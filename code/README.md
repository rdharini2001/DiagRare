# DIAGRARE-BENCH code

These scripts generate diagnostic cases, run open-weight model inference, fit the ranking model, and analyze the evidence and prevalence experiments. General-purpose and medically specialized checkpoints use the same scoring code.

## Core analyses

- `analysis/plackett_luce.py`: fit the Plackett--Luce rank model.
- `analysis/fit_per_all_models.py`: estimate prior and evidence coefficients.
- `analysis/score_predictions.py`: measure diagnostic accuracy and rankings.
- `analysis/make_publication_figures.py`: produce publication figures from the result tables.

## Cases and inference

- `benchmark_generation/generate_vignettes.py`: generate structured cases from the disease ontology.
- `benchmark_generation/new_diseases.py`: additional disease profiles.
- `inference/run_inference_vllm.py`: inference with vLLM.
- `inference/run_model_all_conditions.py`: baseline and intervention runs.
- `inference/prompts.py`: diagnostic prompts and output formats.

## Evidence interventions

- `theory/causal_intervention_grid.py`: 3-by-3 evidence and stated-prevalence experiment.
- `theory/analyze_causal_grid.py`: intervention effects.
- `theory/causal_response_surface.py`: response surfaces and model comparisons.
- `theory/build_sequential_anchoring_cases.py`: sequential case construction.
- `theory/analyze_sequential_anchoring.py`: recovery and order effects.

## External case reports

`theory/external_validation/` contains the CUPCase and RareArena preparation and evaluation scripts. The original licenses continue to govern the third-party data.

## Other analyses

`theory/robustness/` contains alternative choice models, candidate-set tests, clustered uncertainty, feature-shuffling controls, and subgroup comparisons. The `finetuning/` directory contains the QLoRA adaptation scripts. `analysis/self_consistency_analysis.py` covers multi-sample aggregation; `theory/analyze_qwen3_thinking_budget.py` covers reasoning-token budgets.

Run `python code/analysis/make_publication_figures.py` from the project root to reproduce figures from the available result tables. Some end-to-end runs also require the raw predictions or third-party cases noted in the archive README.
