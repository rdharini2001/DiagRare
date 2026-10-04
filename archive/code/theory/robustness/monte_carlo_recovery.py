#!/usr/bin/env python3
"""Monte Carlo recovery study for the Plackett-Luce (gamma_prior, gamma_evidence)
estimator: generate rankings from KNOWN parameters and check that MLE fitting
recovers them, across a grid of (true params) x (sample size n) x (ranking
depth k) x (candidate count N) x (noise level). >=100 replicates per cell.

Reports, per cell: bias, RMSE, Wald-CI coverage (95%, via a numerical Hessian
at the MLE -- cheap because the vectorized NLL makes finite-difference second
derivatives fast), sign-recovery rate (did we get the right sign on both
coefficients?), PER (ratio) recovery, and failure rate (non-convergence).

any real-data result: does the machinery even work when the truth is known?
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from plackett_luce import Choice, ChoiceBatch, neg_log_likelihood_fast, fit_plackett_luce

ROOT = Path(__file__).resolve().parents[3]
RNG_SEED = 20260912


def simulate_ranking_v2(gamma_prior: float, gamma_evidence: float, N: int, k: int,
                         noise: float, rng: np.random.Generator) -> Choice:
    """Draw N candidates' covariates, then sample a top-k Plackett-Luce ranking
    from the resulting utilities. `noise` in [0,1]: probability of picking
    uniformly at random at each step instead of via the PL softmax (models
    ranking noise / bounded rationality)."""
    x_prior = rng.uniform(-9, -0.5, size=N)
    x_evidence = rng.uniform(-5, 5, size=N)
    u = gamma_prior * x_prior + gamma_evidence * x_evidence

    available_mask = np.ones(N, dtype=bool)
    chosen_indices = []
    for _ in range(k):
        avail_idx = np.where(available_mask)[0]
        if rng.random() < noise:
            pick = rng.choice(avail_idx)
        else:
            u_avail = u[avail_idx]
            p = np.exp(u_avail - u_avail.max())
            p /= p.sum()
            pick = rng.choice(avail_idx, p=p)
        chosen_indices.append(int(pick))
        available_mask[pick] = False
    return Choice(chosen=chosen_indices, log_prevalence=x_prior, evidence=x_evidence)


def numerical_hessian(f, x0, eps=1e-3):
    n = len(x0)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp, xpm, xmp, xmm = x0.copy(), x0.copy(), x0.copy(), x0.copy()
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
    return H


def fit_with_wald_ci(choices: list[Choice]) -> dict:
    batch = ChoiceBatch(choices)
    fit = fit_plackett_luce(choices, n_starts=2)
    theta_hat = np.array([fit["gamma0"], fit["gamma_prior"], fit["gamma_evidence"]])
    try:
        H = numerical_hessian(lambda th: neg_log_likelihood_fast(th, batch), theta_hat)
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.where(diag >= 0, diag, np.nan))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan, np.nan])
    return {**fit, "se_gamma_prior": se[1], "se_gamma_evidence": se[2]}


def run_grid(n_replicates: int = 100) -> pd.DataFrame:
    true_params = {
        "null": (0.0, 0.0), "weak_both": (0.3, 0.3), "balanced": (1.0, 1.0),
        "evidence_dominant": (0.2, 1.5), "prior_dominant": (1.5, 0.2),
        "mismatched_sign": (-0.5, 1.0),
    }
    ns = [50, 100, 250, 500, 1000]
    ks = [1, 3, 5]
    Ns = [20, 82]
    noises = [0.0, 0.2]

    rows = []
    rng_master = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    total_cells = len(true_params) * len(ns) * len(ks) * len(Ns) * len(noises)
    cell_i = 0

    for param_name, (gp_true, ge_true) in true_params.items():
        for n in ns:
            for k in ks:
                for N in Ns:
                    for noise in noises:
                        cell_i += 1
                        gps, ges, converged_flags = [], [], []
                        cov_prior, cov_evidence, cov_per = [], [], []
                        for rep in range(n_replicates):
                            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                            choices = [simulate_ranking_v2(gp_true, ge_true, N, k, noise, rng) for _ in range(n)]
                            fit = fit_with_wald_ci(choices)
                            if not fit["converged"]:
                                converged_flags.append(False)
                                continue
                            converged_flags.append(True)
                            gps.append(fit["gamma_prior"])
                            ges.append(fit["gamma_evidence"])
                            if np.isfinite(fit["se_gamma_prior"]) and fit["se_gamma_prior"] > 0:
                                lo, hi = fit["gamma_prior"] - 1.96 * fit["se_gamma_prior"], fit["gamma_prior"] + 1.96 * fit["se_gamma_prior"]
                                cov_prior.append(lo <= gp_true <= hi)
                            if np.isfinite(fit["se_gamma_evidence"]) and fit["se_gamma_evidence"] > 0:
                                lo, hi = fit["gamma_evidence"] - 1.96 * fit["se_gamma_evidence"], fit["gamma_evidence"] + 1.96 * fit["se_gamma_evidence"]
                                cov_evidence.append(lo <= ge_true <= hi)

                        gps, ges = np.array(gps), np.array(ges)
                        if len(gps) == 0:
                            continue
                        per_true = gp_true / ge_true if ge_true != 0 else np.inf
                        per_hat = gps / np.where(ges != 0, ges, np.nan)

                        rows.append({
                            "param_setting": param_name, "gamma_prior_true": gp_true, "gamma_evidence_true": ge_true,
                            "n": n, "k": k, "N_candidates": N, "noise": noise,
                            "n_replicates": n_replicates, "failure_rate": 1 - np.mean(converged_flags),
                            "bias_gamma_prior": np.mean(gps - gp_true), "bias_gamma_evidence": np.mean(ges - ge_true),
                            "rmse_gamma_prior": np.sqrt(np.mean((gps - gp_true) ** 2)),
                            "rmse_gamma_evidence": np.sqrt(np.mean((ges - ge_true) ** 2)),
                            "sign_recovery_prior": np.mean(np.sign(gps) == np.sign(gp_true)) if gp_true != 0 else np.nan,
                            "sign_recovery_evidence": np.mean(np.sign(ges) == np.sign(ge_true)) if ge_true != 0 else np.nan,
                            "ci_coverage_prior": np.mean(cov_prior) if cov_prior else np.nan,
                            "ci_coverage_evidence": np.mean(cov_evidence) if cov_evidence else np.nan,
                            "per_bias": np.nanmean(per_hat - per_true) if np.isfinite(per_true) else np.nan,
                        })
                        if cell_i % 20 == 0:
                            print(f"[{time.time()-t0:.0f}s] cell {cell_i}/{total_cells} "
                                  f"({param_name}, n={n}, k={k}, N={N}, noise={noise})", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = run_grid(n_replicates=100)
    out_dir = ROOT / "results" / "theory" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "monte_carlo_recovery.csv", index=False)
    print(f"\nWrote {out_dir / 'monte_carlo_recovery.csv'} ({len(out)} rows)")
    print("\nSummary at n=1000, k=3, N=82, noise=0 (best-case cell):")
    best = out[(out.n == 1000) & (out.k == 3) & (out.N_candidates == 82) & (out.noise == 0)]
    pd.set_option("display.width", 200)
    print(best[["param_setting", "bias_gamma_prior", "bias_gamma_evidence", "rmse_gamma_prior",
                "rmse_gamma_evidence", "ci_coverage_prior", "ci_coverage_evidence",
                "sign_recovery_prior", "sign_recovery_evidence"]].to_string(index=False))
