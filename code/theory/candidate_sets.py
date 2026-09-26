"""Builds a clinically realistic forced-choice candidate set per vignette:
every disease in the SAME organ system as the true disease (a genuine
differential diagnosis), plus the single highest-prevalence disease in the
whole ontology as a "common alternative" if it isn't already included.
Kept small (typically 6-13 candidates) so direct log-likelihood scoring of
every candidate is affordable, unlike scoring against the full 58-82 item list.
"""
from __future__ import annotations

import pandas as pd


def build_candidate_sets(vignettes: pd.DataFrame, ontology: pd.DataFrame) -> dict[int, list[str]]:
    global_common = ontology.loc[ontology["prevalence"].idxmax(), "disease"]
    by_system = {sys: grp["disease"].tolist() for sys, grp in ontology.groupby("organ_system")}

    out = {}
    for _, v in vignettes.iterrows():
        cands = list(by_system.get(v["organ_system"], [v["target_disease"]]))
        if v["target_disease"] not in cands:
            cands.append(v["target_disease"])
        if global_common not in cands:
            cands.append(global_common)
        out[v["vignette_id"]] = cands
    return out
