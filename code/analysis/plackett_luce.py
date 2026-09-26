"""Plackett-Luce discrete-choice model for LLM top-k diagnostic rankings.

Framing: at each vignette, the model is shown N candidate diseases and asked
to rank its top-k=1..3. We treat this as a sequential Plackett-Luce process
(McFadden-style random-utility choice, generalized to partial top-k rankings,
Plackett 1975 / Luce 1959): each pick is a softmax choice over the remaining
alternatives, so the log-likelihood of an observed top-k ranking (i_1,...,i_k)
out of N alternatives with utilities u is

    sum_{j=1}^{k} [ u_{i_j} - logsumexp( u_m : m in {i_j,...,i_N} \\ {i_1,...,i_{j-1}} ) ]

Utility is linear in two covariates taken directly from the Bayes oracle
(theory/bayes_oracle.py), so that the TRUE Bayes-optimal decision rule is
EXACTLY the gamma_prior = gamma_evidence = 1 member of this model family
(because bayes_log_posterior = log_prevalence + evidence_loglik_ratio + const):

    u(d) = gamma0 + gamma_prior * log_prevalence(d) + gamma_evidence * evidence_loglik_ratio(d)

PER (Prior-Evidence Ratio) = gamma_prior / gamma_evidence:
    PER = 1  -> Bayes-consistent use of population prevalence relative to evidence
    PER > 1  -> excess reliance on prevalence relative to the evidence actually shown
                (the "prevalence-driven collapse" failure mode)
    PER < 1  -> under-weights prevalence relative to a Bayes-optimal reasoner
                (can happen after aggressive debiasing / verifier correction)

Fitting uses only PARSEABLE choice occasions (predictions matching a known
disease name); the fraction used is reported per (model, condition) since a
model that mostly fails to format output (e.g. BioMistral, Qwen2.5-0.5B)
yields sparse, high-variance estimates -- this is made explicit via
bootstrap confidence intervals (resampling vignettes with replacement).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class Choice:
    """One ranking observation: `chosen` is an ordered list of candidate
    indices (into the shared `alternatives` array for this occasion),
    length 1-3; `alternatives` gives every candidate's covariate row."""
    chosen: list[int]
    log_prevalence: np.ndarray   # (N,)
    evidence: np.ndarray         # (N,)


L2_PENALTY = 1e-3  # light ridge on (gamma_prior, gamma_evidence); keeps near-deterministic
                    # rankings (e.g. a model at ~100% accuracy) numerically well-posed --
                    # without it the likelihood is flat along any ray through a perfectly
                    # explanatory direction and gamma magnitudes diverge (PER, the ratio,
                    # is unaffected either way -- see the oracle sanity check).


def neg_log_likelihood(theta: np.ndarray, choices: list[Choice]) -> float:
    """Reference (slow, O(M) Python loop) implementation -- kept only as the
    ground truth that the vectorized `neg_log_likelihood_fast` is checked
    against; not used in fitting once that check passes."""
    gamma0, gamma_prior, gamma_evidence = theta
    total = 0.0
    for c in choices:
        u = gamma0 + gamma_prior * c.log_prevalence + gamma_evidence * c.evidence
        remaining = list(range(len(u)))
        for picked in c.chosen:
            total += u[picked] - logsumexp(u[remaining])
            remaining.remove(picked)
    penalty = L2_PENALTY * (gamma_prior ** 2 + gamma_evidence ** 2)
    return -total + penalty


class ChoiceBatch:
    """Stacks a list of Choice objects into matrices so the Plackett-Luce
    log-likelihood can be evaluated with a handful of vectorized numpy ops
    instead of an O(M) Python loop -- this is the difference between a single
    fit taking ~20s and ~milliseconds, which matters once we need hundreds of
    bootstrap refits per (model, condition).

    Choices may have DIFFERENT candidate-set sizes N_i (e.g. same-organ-system
    restriction gives variable-size candidate sets per vignette) -- shorter
    ones are padded to the batch max with a `valid` mask so padding positions
    are excluded from the softmax denominator (equivalent to -inf utility)."""

    def __init__(self, choices: list[Choice]):
        self.M = len(choices)
        self.N = max((len(c.log_prevalence) for c in choices), default=0)
        self.log_prevalence = np.zeros((self.M, self.N))
        self.evidence = np.zeros((self.M, self.N))
        self.valid = np.zeros((self.M, self.N), dtype=bool)
        for i, c in enumerate(choices):
            n_i = len(c.log_prevalence)
            self.log_prevalence[i, :n_i] = c.log_prevalence
            self.evidence[i, :n_i] = c.evidence
            self.valid[i, :n_i] = True
        self.max_k = max((len(c.chosen) for c in choices), default=0)
        # picked[r, i] = index chosen at round r for choice i (0 if choice has no round-r pick)
        # active[r, i] = whether choice i actually has a round-r pick
        self.picked = np.zeros((self.max_k, self.M), dtype=int)
        self.active = np.zeros((self.max_k, self.M), dtype=bool)
        for i, c in enumerate(choices):
            for r, idx in enumerate(c.chosen):
                self.picked[r, i] = idx
                self.active[r, i] = True


def neg_log_likelihood_fast(theta: np.ndarray, batch: ChoiceBatch) -> float:
    gamma0, gamma_prior, gamma_evidence = theta
    if batch.M == 0:
        return 0.0
    u = gamma0 + gamma_prior * batch.log_prevalence + gamma_evidence * batch.evidence  # (M, N)
    mask = batch.valid.copy()
    total = 0.0
    rows = np.arange(batch.M)
    for r in range(batch.max_k):
        active = batch.active[r]
        u_masked = np.where(mask, u, -np.inf)
        logZ = logsumexp(u_masked, axis=1)  # (M,)
        picked_idx = batch.picked[r]
        u_picked = u[rows, picked_idx]
        contrib = np.where(active, u_picked - logZ, 0.0)
        total += contrib.sum()
        # remove the picked alternative from future rounds (only where active)
        mask[rows[active], picked_idx[active]] = False
    penalty = L2_PENALTY * (gamma_prior ** 2 + gamma_evidence ** 2)
    return -total + penalty


def fit_plackett_luce(choices: list[Choice], n_starts: int = 3, seed: int = 0) -> dict:
    if len(choices) == 0:
        return {"gamma0": np.nan, "gamma_prior": np.nan, "gamma_evidence": np.nan,
                "PER": np.nan, "n": 0, "converged": False}

    batch = ChoiceBatch(choices)
    rng = np.random.default_rng(seed)
    best = None
    for i in range(n_starts):
        x0 = np.zeros(3) if i == 0 else rng.normal(0, 0.5, size=3)
        res = minimize(neg_log_likelihood_fast, x0, args=(batch,), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res

    gamma0, gamma_prior, gamma_evidence = best.x
    per = gamma_prior / gamma_evidence if abs(gamma_evidence) > 1e-8 else np.inf
    return {"gamma0": gamma0, "gamma_prior": gamma_prior, "gamma_evidence": gamma_evidence,
            "PER": per, "n": len(choices), "converged": bool(best.success), "nll": best.fun}


def bootstrap_samples(choices: list[Choice], n_boot: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """Raw bootstrap draws (resampling vignettes/choice-occasions with
    replacement) -- the building block for both bootstrap_ci (percentiles)
    and pairwise significance tests between two models/conditions (paired or
    unpaired difference-of-bootstrap-distributions test)."""
    if len(choices) < 10:
        return {"PER": np.array([]), "gamma_prior": np.array([]), "gamma_evidence": np.array([])}
    rng = np.random.default_rng(seed)
    n = len(choices)
    pers, gps, ges = [], [], []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [choices[i] for i in idx]
        fit = fit_plackett_luce(sample, n_starts=1, seed=seed + b)
        if fit["converged"] and np.isfinite(fit["PER"]):
            pers.append(fit["PER"])
            gps.append(fit["gamma_prior"])
            ges.append(fit["gamma_evidence"])
    return {"PER": np.array(pers), "gamma_prior": np.array(gps), "gamma_evidence": np.array(ges)}


def bootstrap_ci(choices: list[Choice], n_boot: int = 200, seed: int = 0) -> dict:
    samples = bootstrap_samples(choices, n_boot=n_boot, seed=seed)
    pers, gps, ges = samples["PER"], samples["gamma_prior"], samples["gamma_evidence"]
    if len(pers) == 0:
        return {"PER_lo": np.nan, "PER_hi": np.nan, "gamma_prior_lo": np.nan, "gamma_prior_hi": np.nan,
                "gamma_evidence_lo": np.nan, "gamma_evidence_hi": np.nan}
    return {
        "PER_lo": np.percentile(pers, 2.5), "PER_hi": np.percentile(pers, 97.5),
        "gamma_prior_lo": np.percentile(gps, 2.5), "gamma_prior_hi": np.percentile(gps, 97.5),
        "gamma_evidence_lo": np.percentile(ges, 2.5), "gamma_evidence_hi": np.percentile(ges, 97.5),
    }


def build_choices_from_predictions(
    predictions: pd.DataFrame, oracle_table: pd.DataFrame, vignette_ids: set | None = None
) -> tuple[list[Choice], int, int]:
    """predictions must have columns vignette_id, prediction_1, prediction_2, prediction_3.
    oracle_table must have columns vignette_id, disease, log_prevalence, evidence_loglik_ratio
    (i.e. theory/bayes_oracle.py's output, giving the FULL candidate set shown for that vignette).
    Returns (choices, n_total_occasions, n_used_occasions)."""
    oracle_by_vid = {vid: grp.reset_index(drop=True) for vid, grp in oracle_table.groupby("vignette_id")}
    choices = []
    n_total = 0
    n_used = 0
    for _, row in predictions.iterrows():
        vid = row["vignette_id"]
        if vignette_ids is not None and vid not in vignette_ids:
            continue
        n_total += 1
        grp = oracle_by_vid.get(vid)
        if grp is None:
            continue
        disease_to_idx = {d: i for i, d in enumerate(grp["disease"])}
        log_prev = grp["log_prevalence"].to_numpy()
        evidence = grp["evidence_loglik_ratio"].to_numpy()

        chosen = []
        seen = set()
        for col in ("prediction_1", "prediction_2", "prediction_3"):
            val = str(row.get(col, "")).strip()
            if val in disease_to_idx and val not in seen:
                chosen.append(disease_to_idx[val])
                seen.add(val)
            else:
                break  # stop at first unparseable/duplicate entry
        if not chosen:
            continue
        n_used += 1
        choices.append(Choice(chosen=chosen, log_prevalence=log_prev, evidence=evidence))
    return choices, n_total, n_used
