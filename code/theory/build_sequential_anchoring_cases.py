#!/usr/bin/env python3
"""Construct the sequential diagnostic-revision experiment.

For each target and same-organ competing diagnosis, the same two evidence blocks are
presented in opposite orders. The final evidence is identical across orders, which
allows recovery after an initial competing diagnosis and final-step order dependence
to be measured separately. Stated prevalence is held equal throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from causal_intervention_grid import pick_confounder  # noqa: E402
from symbolic_verifier import split_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EQUAL_PREV = 1000  # per 100k -- neutral, identical stated prevalence for both diseases


def build_findings(row: pd.Series) -> str:
    """Same finding-selection rule as causal_intervention_grid.py's 'strong'
    evidence level: all required findings + top-2 supporting findings."""
    req = sorted(split_tokens(row["required"]))
    sup = sorted(split_tokens(row["supporting"]))
    return "; ".join(req + sup[:2])


def main() -> None:
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")

    rows = []
    pair_id = 0
    for _, target_row in ontology.iterrows():
        confounder_row = pick_confounder(target_row, ontology)
        if confounder_row is None or len(split_tokens(target_row["required"])) < 2:
            continue
        target_findings = build_findings(target_row)
        confounder_findings = build_findings(confounder_row)
        if not target_findings or not confounder_findings:
            continue

        for order in ["confounder_first", "target_first"]:
            first_findings = confounder_findings if order == "confounder_first" else target_findings
            second_new_findings = target_findings if order == "confounder_first" else confounder_findings
            rows.append({
                "pair_id": pair_id, "target_disease": target_row["disease"],
                "confounder_disease": confounder_row["disease"], "organ_system": target_row["organ_system"],
                "order": order, "step": 1,
                "cumulative_findings": first_findings,
                "target_prevalence_per_100k": EQUAL_PREV, "confounder_prevalence_per_100k": EQUAL_PREV,
            })
            seen = set()
            combined = []
            for f in first_findings.split("; ") + second_new_findings.split("; "):
                if f not in seen:
                    seen.add(f)
                    combined.append(f)
            rows.append({
                "pair_id": pair_id, "target_disease": target_row["disease"],
                "confounder_disease": confounder_row["disease"], "organ_system": target_row["organ_system"],
                "order": order, "step": 2,
                "cumulative_findings": "; ".join(combined),
                "initial_findings": first_findings, "new_findings": second_new_findings,
                "target_prevalence_per_100k": EQUAL_PREV, "confounder_prevalence_per_100k": EQUAL_PREV,
            })
        pair_id += 1

    out = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "causal_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sequential_anchoring_cases.csv"
    out.to_csv(out_path, index=False)
    print(f"Built {pair_id} (target, confounder) pairs x 2 orders x 2 steps = {len(out)} rows")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
