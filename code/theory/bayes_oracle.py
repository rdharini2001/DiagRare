"""Compute prior and evidence features from the disease-finding ontology."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def split_tokens(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return set()
    return {t.strip() for t in str(value).split(";") if t.strip()}


@dataclass(frozen=True)
class OracleParams:
    p_req: float = 0.90   # P(finding present | it is a REQUIRED finding of the disease)
    p_sup: float = 0.50   # P(finding present | it is a SUPPORTING finding of the disease)
    p_bg: float = 0.03    # P(finding present | it is unrelated to the disease -- background rate)
    p_exc: float = 0.02   # P(finding present | it is an EXCLUDING finding of the disease)


def finding_class(disease_row: pd.Series, finding: str) -> str:
    if finding in split_tokens(disease_row["required"]):
        return "req"
    if finding in split_tokens(disease_row["supporting"]):
        return "sup"
    if finding in split_tokens(disease_row["excluding"]):
        return "exc"
    return "bg"


def log_likelihood_evidence_given_disease(
    positive: set[str], negative: set[str], disease_row: pd.Series, params: OracleParams
) -> float:
    """log P(evidence | D) under the Naive-Bayes finding-emission model, restricted to
    the finding vocabulary actually mentioned in this vignette (present or explicitly
    absent) -- findings never mentioned are marginalized out (contribute no information)."""
    p_of = {"req": params.p_req, "sup": params.p_sup, "exc": params.p_exc, "bg": params.p_bg}
    ll = 0.0
    for f in positive:
        p = p_of[finding_class(disease_row, f)]
        ll += np.log(p)
    for f in negative:
        p = p_of[finding_class(disease_row, f)]
        ll += np.log(1.0 - p)
    return ll


def compute_oracle_table(vignettes: pd.DataFrame, ontology: pd.DataFrame, params: OracleParams) -> pd.DataFrame:
    """Long-format table: one row per (vignette, candidate disease) with the
    Bayes log-posterior, the evidence-only log-likelihood-ratio (vs. a
    finding-blind disease -- i.e. excess support beyond chance), and prevalence.
    `candidate disease` ranges over every disease in `ontology` (the full
    allowed list for that vignette's benchmark version)."""
    ontology = ontology.set_index("disease", drop=False)
    log_prev = {d: np.log(max(row["prevalence"], 1e-8)) for d, row in ontology.iterrows()}
    diseases = list(ontology.index)

    rows = []
    for _, v in vignettes.iterrows():
        pos = split_tokens(v["positive_findings"])
        neg = split_tokens(v["negative_findings"])
        n_mentioned = len(pos) + len(neg)
        # "finding-blind" reference likelihood: every mentioned finding treated as
        # background-rate evidence -- the log-likelihood ratio below isolates how
        # much MORE (or less) likely this evidence is under D specifically.
        blind_ll = sum(np.log(params.p_bg) for _ in pos) + sum(np.log(1 - params.p_bg) for _ in neg)

        ll_by_disease = {}
        for d in diseases:
            drow = ontology.loc[d]
            ll_by_disease[d] = log_likelihood_evidence_given_disease(pos, neg, drow, params)

        log_post_unnorm = {d: log_prev[d] + ll_by_disease[d] for d in diseases}
        m = max(log_post_unnorm.values())
        z = np.log(sum(np.exp(lp - m) for lp in log_post_unnorm.values())) + m

        for d in diseases:
            rows.append({
                "vignette_id": v["vignette_id"],
                "disease": d,
                "is_true_disease": d == v["target_disease"],
                "log_prevalence": log_prev[d],
                "evidence_loglik_ratio": ll_by_disease[d] - blind_ll,
                "bayes_log_posterior": log_post_unnorm[d] - z,
                "n_mentioned": n_mentioned,
            })
    return pd.DataFrame(rows)


def oracle_accuracy(oracle_table: pd.DataFrame, vignettes: pd.DataFrame, k: int = 1) -> float:
    """Bayes-optimal achievable top-k accuracy: rank candidates by posterior
    within each vignette, check if the true disease is in the top k."""
    correct = 0
    for vid, grp in oracle_table.groupby("vignette_id"):
        top_k = grp.nlargest(k, "bayes_log_posterior")
        correct += int(top_k["is_true_disease"].any())
    return correct / vignettes["vignette_id"].nunique()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    vignettes = pd.read_csv(root / "data" / "expanded" / "vignettes_combined.csv")
    ontology = pd.read_csv(root / "data" / "expanded" / "ontology_diseases_expanded.csv")
    params = OracleParams()

    table = compute_oracle_table(vignettes, ontology, params)
    out_dir = root / "results" / "theory"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "bayes_oracle_table_expanded.csv", index=False)

    print("Bayes-optimal oracle (Naive-Bayes finding-emission model)")
    print(f"  params: p_req={params.p_req} p_sup={params.p_sup} p_bg={params.p_bg} p_exc={params.p_exc}")
    for k in (1, 3):
        acc = oracle_accuracy(table, vignettes, k=k)
        print(f"  Bayes-optimal Top-{k} accuracy (ALL splits): {100*acc:.1f}%")

    for split_name in vignettes["split"].unique():
        vids = set(vignettes.loc[vignettes["split"] == split_name, "vignette_id"])
        sub = table[table["vignette_id"].isin(vids)]
        acc1 = oracle_accuracy(sub, vignettes[vignettes["split"] == split_name], k=1)
        acc3 = oracle_accuracy(sub, vignettes[vignettes["split"] == split_name], k=3)
        print(f"  [{split_name}] Bayes-optimal Top-1={100*acc1:.1f}% Top-3={100*acc3:.1f}%")

    for tier in vignettes["difficulty"].dropna().unique():
        vids = set(vignettes.loc[vignettes["difficulty"] == tier, "vignette_id"])
        sub = table[table["vignette_id"].isin(vids)]
        acc1 = oracle_accuracy(sub, vignettes[vignettes["difficulty"] == tier], k=1)
        print(f"  [difficulty={tier}] Bayes-optimal Top-1={100*acc1:.1f}%")

    print(f"\nWrote {out_dir / 'bayes_oracle_table_expanded.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
