#!/usr/bin/env python3
"""The single most important missing experiment (per external review): a
factorial causal intervention grid that independently manipulates ONLY the
stated prior and ONLY the diagnostic evidence, holding everything else fixed,
for a clean 2-candidate forced choice (target disease vs. its nearest
same-organ-system confounder).

Design (3x3, ~27 disease pairs = 243 combinations):
  - Evidence axis (weak/medium/strong): how many of the target disease's own
    required/supporting findings are shown, and whether the confounder's
    findings are also present or explicitly excluded.
  - Prior axis (target-rare/equal/target-common): an EXPLICIT NUMERIC
    prevalence statement for both diseases in the prompt, deliberately
    decoupled from real epidemiology (can be the reverse of a disease's true
    rarity) -- so any effect is attributable to the model updating on the
    STATED number, not recalling memorized real-world prevalence.

Each combination is one prompt with EXACTLY 2 valid answers, making parsing
trivial and the outcome a clean binary (chose target vs. chose confounder).

The critical downstream test (done after inference, in
analyze_causal_grid.py): does each model's ALREADY-FITTED gamma_prior (from
the observational Plackett-Luce fit on free-text baseline predictions)
predict that SAME model's empirically measured sensitivity to this
experimental prior manipulation? Does gamma_evidence predict sensitivity to
the evidence manipulation? That is a behaviorally-validated claim, not just
a correlational one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from symbolic_verifier import split_tokens

ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_LEVELS = ["weak", "medium", "strong"]
PRIOR_LEVELS = ["target_rare", "equal", "target_common"]
PRIOR_NUMERIC = {  # (target_prevalence_per_100k, confounder_prevalence_per_100k)
    "target_rare": (1, 10000), "equal": (1000, 1000), "target_common": (10000, 1),
}


def pick_confounder(target_row: pd.Series, ontology: pd.DataFrame) -> pd.Series | None:
    same_system = ontology[(ontology.organ_system == target_row.organ_system) & (ontology.disease != target_row.disease)]
    if same_system.empty:
        return None
    return same_system.sort_values("prevalence", ascending=False).iloc[0]


def build_evidence(target_row: pd.Series, confounder_row: pd.Series, level: str, rng) -> tuple[str, str]:
    """Returns (positive_findings, negative_findings) strings for this evidence level.
    Evidence always concerns the TARGET disease's own finding vocabulary."""
    req = split_tokens(target_row["required"])
    sup = split_tokens(target_row["supporting"])
    comp_req = split_tokens(confounder_row["required"])
    comp_sup = split_tokens(confounder_row["supporting"])

    req, sup = sorted(req), sorted(sup)
    if level == "weak":
        pos = req[:1]
        neg = []
    elif level == "medium":
        pos = req[:max(1, len(req) // 2)] + sup[:1]
        neg = []
    else:  # strong
        pos = req + sup[:2]
        # explicitly exclude the confounder's own distinguishing required findings --
        # but never a finding already asserted PRESENT above (e.g. "dyspnea" can be a
        # supporting finding of the target AND a required finding of the confounder;
        # asserting it both present and absent would be a direct self-contradiction).
        candidate_neg = [f for f in sorted(comp_req) if f not in pos]
        neg = candidate_neg[:2]
    return ";".join(pos), ";".join(neg)


def main() -> None:
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    import numpy as np
    rng = np.random.default_rng(20260912)

    # pick up to 3 target diseases per organ system with a valid same-organ confounder
    targets = []
    for organ, grp in ontology.groupby("organ_system"):
        candidates = grp.sample(frac=1.0, random_state=20260912).to_dict("records")
        picked = 0
        for row in candidates:
            row = pd.Series(row)
            comp = pick_confounder(row, ontology)
            if comp is not None and len(split_tokens(row["required"])) >= 2:
                targets.append((row, comp))
                picked += 1
            if picked >= 3:
                break

    print(f"Selected {len(targets)} (target, confounder) disease pairs across {ontology.organ_system.nunique()} organ systems")

    rows = []
    combo_id = 0
    for target_row, comp_row in targets:
        for evidence_level in EVIDENCE_LEVELS:
            pos, neg = build_evidence(target_row, comp_row, evidence_level, rng)
            for prior_level in PRIOR_LEVELS:
                target_prev, comp_prev = PRIOR_NUMERIC[prior_level]
                combo_id += 1
                rows.append({
                    "combo_id": combo_id,
                    "target_disease": target_row["disease"], "confounder_disease": comp_row["disease"],
                    "organ_system": target_row["organ_system"],
                    "evidence_level": evidence_level, "prior_level": prior_level,
                    "positive_findings": pos, "negative_findings": neg,
                    "target_prevalence_per_100k": target_prev, "confounder_prevalence_per_100k": comp_prev,
                })

    out = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "causal_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "causal_grid.csv", index=False)
    print(f"Wrote {len(out)} combinations to {out_dir / 'causal_grid.csv'}")
    print(out.groupby(["evidence_level", "prior_level"]).size())


if __name__ == "__main__":
    main()
