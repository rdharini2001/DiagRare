#!/bin/bash
# Re-runs the entire downstream analysis sequence after the expanded model
# panel + negative-control inference lands. Every step here is CPU-only and
# dynamically globs its input directory, so it automatically picks up every
# new model's predictions with no code changes.
#
# Usage: ./reproduce_all_analyses.sh [--full]
#   --full  use the full-scale CUPCase (3562) / RareArena (22901) external
#           validation results instead of the 250-case subsets. MedQA was
#           dropped from the benchmark entirely (see data_final/DATASHEET.md
#           for why -- its TF-IDF evidence axis scored below chance even at
#           full scale, a structural property of board-exam writing).
set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
FULL_FLAG="$1"

echo "=== 1/10: score_predictions.py (accuracy summary, all models) ==="
python3 analysis/score_predictions.py \
  --pred_dir results/predictions \
  --vignettes data/expanded/vignettes_combined.csv \
  --ontology data/expanded/ontology_diseases_expanded.csv \
  --out results/open_weight_summary.csv

echo "=== 2/10: fit_per_all_models.py (gamma_prior/gamma_evidence, all models) ==="
python3 analysis/fit_per_all_models.py

echo "=== 3/10: analyze_causal_grid.py (decisive result, expanded panel) ==="
python3 theory/analyze_causal_grid.py

echo "=== 4/10: causal_validation_jackknife.py ==="
python3 theory/robustness/causal_validation_jackknife.py

echo "=== 5/10: analyze_cupcase.py + analyze_rarearena.py (external validation, expanded panel) ==="
python3 theory/external_validation/analyze_cupcase.py $FULL_FLAG
python3 theory/external_validation/analyze_rarearena.py $FULL_FLAG

echo "=== 6/10: clinical_stakes_simulation.py ==="
python3 theory/clinical_stakes_simulation.py

python3 theory/robustness/model_panel_regression.py

echo "=== 8/10: finetune_recipe_comparison.py (Mistral-family recipe comparison) ==="
python3 theory/robustness/finetune_recipe_comparison.py

echo "=== 9/10: analyze_negative_control.py ==="
python3 theory/analyze_negative_control.py

echo "=== 10/10: discordance_index.py + specialization_comparison.py ==="
python3 theory/discordance_index.py $FULL_FLAG
python3 theory/specialization_comparison.py $FULL_FLAG

echo ""
echo "=== ALL DONE. Regenerate figures next: ==="
echo "  python3 analysis/make_theory_figures.py"
echo "  python3 analysis/make_robustness_figures.py"
