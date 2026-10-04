#!/usr/bin/env python3
"""Compute rare-versus-common diagnostic discordance.

Accuracy discordance compares accuracy on common and rare targets within the same
evaluation. Evidence discordance applies the same comparison to subgroup-specific
evidence coefficients when those estimates are available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6


def d_acc(acc_common: float, acc_rare: float) -> float:
    return (acc_common - acc_rare) / (acc_common + acc_rare + EPS)


def diagrare_bench_discordance() -> pd.DataFrame:
    """D_acc on DiagRare-Bench using its native is_rare label, for every model
    with baseline predictions -- the full expanded panel, not just the
    original 8 used in Sec 7.13's matched-pair analysis."""
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    rows = []
    for f in sorted((ROOT / "results" / "predictions").glob("*__baseline.csv")):
        model_tag = f.stem.replace("__baseline", "")
        preds = pd.read_csv(f)
        m = vignettes.merge(preds, on="vignette_id", how="inner")
        m["correct"] = m["prediction_1"].astype(str).str.strip() == m["target_disease"].astype(str).str.strip()
        acc_common = m.loc[~m["is_rare"], "correct"].mean()
        acc_rare = m.loc[m["is_rare"], "correct"].mean()
        rows.append({"model": model_tag, "dataset": "DiagRare-Bench",
                     "acc_common": acc_common, "acc_rare": acc_rare,
                     "n_common": (~m["is_rare"]).sum(), "n_rare": m["is_rare"].sum(),
                     "D_acc": d_acc(acc_common, acc_rare)})
    return pd.DataFrame(rows)


def real_dataset_discordance_adjusted(name: str, pred_dir: Path, data_csv: Path) -> pd.DataFrame:
    """Evidence-strength-ADJUSTED version of real_dataset_discordance.

    Discovered empirically (not assumed): in case-report corpora, the
    "common disease" tercile has systematically WEAKER textual evidence
    quality than the "rare disease" tercile (e.g. on CUPCase, the fraction
    of cases where the correct diagnosis has the single highest TF-IDF
    match is 77.1% in the rare tercile vs. only 36.9% in the common
    tercile, barely above the 25% random baseline) -- almost certainly
    because case reports about a well-known common disease get published
    for some OTHER interesting reason (an atypical complication, an
    unusual context) and so do not emphasize disease-defining features the
    way a report specifically documenting a rare condition does. This
    means a raw common-vs-rare accuracy gap in a case-report corpus
    conflates "harder because rare" with "harder because the write-up is
    less evidentially clear" -- exactly the confound DiagRare-Bench's matched
    construction (Sec 7.13) rules out by design.

    Fix: logistic regression of is_correct on a common-vs-rare indicator
    AND the case's own TF-IDF evidence-match strength (for the correct
    diagnosis) as a covariate. The coefficient on the rarity indicator,
    after this adjustment, estimates the same rare/common gap DiagRare-Bench's
    matching estimates directly -- holding evidence quality fixed instead
    of holding it fixed by construction.
    """
    import statsmodels.api as sm

    if not data_csv.exists() or not pred_dir.exists():
        return pd.DataFrame()
    data = pd.read_csv(data_csv)
    data["log_pubmed"] = np.log(data["pubmed_count_correct"] + 1)
    terciles = data["log_pubmed"].quantile([1 / 3, 2 / 3]).to_numpy()
    data["rarity_group"] = pd.cut(data["log_pubmed"], bins=[-np.inf, terciles[0], terciles[1], np.inf],
                                    labels=["rare", "mid", "common"])

    rows = []
    for f in sorted(pred_dir.glob("*.csv")):
        model_tag = f.stem
        preds = pd.read_csv(f)
        m = data.merge(preds, on="case_id", how="inner")
        m = m[m.rarity_group.isin(["rare", "common"])].copy()
        if len(m) < 30 or m["is_correct"].nunique() < 2:
            continue
        m["is_common"] = (m.rarity_group == "common").astype(int)
        X = sm.add_constant(m[["is_common", "evidence_tfidf_correct_diagnosis"]])
        y = m["is_correct"].astype(int)
        try:
            fit = sm.Logit(y, X).fit(disp=0)
            coef_common = fit.params["is_common"]
            p_common = fit.pvalues["is_common"]
        except Exception:
            coef_common, p_common = np.nan, np.nan
        rows.append({"model": model_tag, "dataset": name, "n": len(m),
                     "adjusted_common_coef": coef_common, "adjusted_common_p": p_common})
    return pd.DataFrame(rows)


def real_dataset_discordance(name: str, pred_dir: Path, data_csv: Path) -> pd.DataFrame:
    """D_acc on a real external dataset (CUPCase/RareArena), splitting
    into common/rare by TERCILE of log(PubMed count) for the correct
    diagnosis -- the same real, externally-sourced prevalence proxy used
    throughout Sec 7.15, applied here as a rarity label rather than a
    continuous prior."""
    if not data_csv.exists() or not pred_dir.exists():
        return pd.DataFrame()
    data = pd.read_csv(data_csv)
    data["log_pubmed"] = np.log(data["pubmed_count_correct"] + 1)
    terciles = data["log_pubmed"].quantile([1 / 3, 2 / 3]).to_numpy()
    data["rarity_group"] = pd.cut(data["log_pubmed"], bins=[-np.inf, terciles[0], terciles[1], np.inf],
                                    labels=["rare", "mid", "common"])

    rows = []
    for f in sorted(pred_dir.glob("*.csv")):
        model_tag = f.stem
        preds = pd.read_csv(f)
        m = data.merge(preds, on="case_id", how="inner")
        common = m[m.rarity_group == "common"]
        rare = m[m.rarity_group == "rare"]
        if len(common) < 10 or len(rare) < 10:
            continue
        acc_common = common["is_correct"].mean()
        acc_rare = rare["is_correct"].mean()
        rows.append({"model": model_tag, "dataset": name,
                     "acc_common": acc_common, "acc_rare": acc_rare,
                     "n_common": len(common), "n_rare": len(rare),
                     "D_acc": d_acc(acc_common, acc_rare)})
    return pd.DataFrame(rows)


def oracle_anchor_check() -> None:
    """Empirically verify the D_ev=0 proposition: apply the SAME matched-pair
    procedure to the Bayes-oracle's own top-3 ranking (not a model's), which
    should recover D_ev approx 0 since the oracle IS the (1,1) Bayes-optimal
    reasoner by construction."""
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    pairs = pd.read_csv(ROOT / "results" / "theory" / "robustness" / "rare_common_matched_pairs.csv")

    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    from plackett_luce import build_choices_from_predictions, fit_plackett_luce

    oracle_preds = []
    for vid, grp in oracle.groupby("vignette_id"):
        top3 = grp.nlargest(3, "bayes_log_posterior")["disease"].tolist()
        while len(top3) < 3:
            top3.append("")
        oracle_preds.append({"vignette_id": vid, "prediction_1": top3[0],
                              "prediction_2": top3[1], "prediction_3": top3[2]})
    oracle_preds = pd.DataFrame(oracle_preds)

    rare_vids = set(pairs["rare_vignette_id"])
    common_vids = set(pairs["common_vignette_id"])
    ge = {}
    for label, vid_set in [("common", common_vids), ("rare", rare_vids)]:
        sub = oracle_preds[oracle_preds.vignette_id.isin(vid_set)]
        choices, _, n_used = build_choices_from_predictions(sub, oracle)
        fit = fit_plackett_luce(choices)
        ge[label] = fit["gamma_evidence"]
    d_ev_oracle = (ge["common"] - ge["rare"]) / (abs(ge["common"]) + abs(ge["rare"]) + EPS)
    print(f"\n=== Oracle anchor check ===")
    print(f"Oracle gamma_evidence: common={ge['common']:.4f}, rare={ge['rare']:.4f}")
    print(f"Oracle D_ev = {d_ev_oracle:.4f} (proposition predicts approx 0)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="use the full-scale CUPCase (3562) / RareArena (22901) external "
                          "validation results instead of the 250-case subsets")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "theory"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_subdir = "external_validation_full" if args.full else "external_validation"
    cupcase_csv = "cupcase_full_with_evidence.csv" if args.full else "cupcase_with_evidence.csv"
    rarearena_csv = "rarearena_full_with_evidence.csv" if args.full else "rarearena_with_evidence.csv"
    out_suffix = "_full" if args.full else ""

    print("=== DiagRare-Bench Discordance (native is_rare label, full panel) ===")
    dx = diagrare_bench_discordance()
    print(dx.sort_values("D_acc", ascending=False).to_string(index=False))

    print("\n=== CUPCase Discordance (PubMed-count tercile split) ===")
    cc = real_dataset_discordance("CUPCase", ROOT / "results" / "theory" / ext_subdir / "cupcase",
                                    ROOT / "data" / "external" / cupcase_csv)
    if len(cc):
        print(cc.sort_values("D_acc", ascending=False).to_string(index=False))

    print("\n=== RareArena Discordance (PubMed-count tercile split) ===")
    ra = real_dataset_discordance("RareArena", ROOT / "results" / "theory" / ext_subdir / "rarearena",
                                    ROOT / "data" / "external" / rarearena_csv)
    if len(ra):
        print(ra.sort_values("D_acc", ascending=False).to_string(index=False))

    print("\n=== Evidence-adjusted rarity coefficient (controls for the case-report evidence-strength confound) ===")
    adj_rows = []
    for name, pred_dir, data_csv in [
        ("CUPCase", ROOT / "results" / "theory" / ext_subdir / "cupcase",
         ROOT / "data" / "external" / cupcase_csv),
        ("RareArena", ROOT / "results" / "theory" / ext_subdir / "rarearena",
         ROOT / "data" / "external" / rarearena_csv),
    ]:
        adj = real_dataset_discordance_adjusted(name, pred_dir, data_csv)
        if len(adj):
            adj_rows.append(adj)
            print(f"\n-- {name} --")
            print(adj.sort_values("adjusted_common_coef", ascending=False).to_string(index=False))
    if adj_rows:
        adj_all = pd.concat(adj_rows, ignore_index=True)
        adj_all.to_csv(out_dir / f"discordance_index_adjusted{out_suffix}.csv", index=False)

    all_d = pd.concat([dx, cc, ra], ignore_index=True)
    out_path = out_dir / f"discordance_index{out_suffix}.csv"
    all_d.to_csv(out_path, index=False)

    # cross-dataset consistency: does a model's DiagRare-Bench D_acc predict its
    # real-dataset D_acc? (analogous to the gamma_evidence cross-dataset checks)
    if len(all_d):
        pivot = all_d.pivot_table(index="model", columns="dataset", values="D_acc")
        print("\n=== Discordance Index by model x dataset ===")
        print(pivot.to_string())
        pivot.to_csv(out_dir / f"discordance_index_pivot{out_suffix}.csv")

    oracle_anchor_check()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
