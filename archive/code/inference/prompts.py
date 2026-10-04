"""Prompt templates for the DiagRare open-weight inference experiments.

All conditions ask for the same output format as the primary evaluation's
evaluation prompt (top-3 diagnoses, exact ontology names) so scoring code
is shared across every condition. Conditions differ only in the framing
around the evidence, isolating the effect of each intervention.
"""
from __future__ import annotations

DISEASE_LIST_HEADER = "DISEASES: {disease_list}"
FINDINGS_BLOCK = "POSITIVE FINDINGS: {positive}\nNEGATIVE/ABSENT FINDINGS: {negative}"
OUTPUT_INSTRUCTIONS = (
    "Respond ONLY with:\n"
    "1. [Disease]\n"
    "2. [Disease]\n"
    "3. [Disease]"
)


def _base_body(disease_list: str, positive: str, negative: str) -> str:
    return f"{DISEASE_LIST_HEADER.format(disease_list=disease_list)}\n\n" \
           f"{FINDINGS_BLOCK.format(positive=positive, negative=negative)}"


def baseline(disease_list: str, positive: str, negative: str) -> str:
    """Matches the primary evaluation's evaluation prompt verbatim in spirit."""
    return (
        "You are a diagnostic reasoning system. Given the clinical findings below, "
        "provide your top 3 most likely diagnoses from this list:\n\n"
        f"{_base_body(disease_list, positive, negative)}\n\n{OUTPUT_INSTRUCTIONS}"
    )


def debias(disease_list: str, positive: str, negative: str) -> str:
    """Explicit instruction to reason only from stated evidence, not population base rates."""
    return (
        "You are a diagnostic reasoning system. Given the clinical findings below, "
        "provide your top 3 most likely diagnoses from this list.\n\n"
        "IMPORTANT: Base your ranking strictly on which diagnoses are logically consistent "
        "with the POSITIVE and NEGATIVE findings stated below. Do NOT let a diagnosis's "
        "general population prevalence outweigh a diagnosis that is directly supported by "
        "the stated findings and not contradicted by any stated absent finding.\n\n"
        f"{_base_body(disease_list, positive, negative)}\n\n{OUTPUT_INSTRUCTIONS}"
    )


def chain_of_thought(disease_list: str, positive: str, negative: str) -> str:
    """CoT: reason step by step, but still emit the required final format."""
    return (
        "You are a diagnostic reasoning system. Given the clinical findings below, "
        "think step by step about which diagnoses are consistent with the POSITIVE findings "
        "and not contradicted by the NEGATIVE/ABSENT findings, then give your top 3 most "
        "likely diagnoses from this list.\n\n"
        f"{_base_body(disease_list, positive, negative)}\n\n"
        "First reason briefly (2-4 sentences) about which candidate diagnoses match the "
        "positive findings and are not ruled out by the negative findings. Then, on new lines, "
        f"{OUTPUT_INSTRUCTIONS}"
    )


def counterfactual_rare_clinic(disease_list: str, positive: str, negative: str) -> str:
    """Reframe the base-rate context: rare-disease referral center."""
    return (
        "You are a diagnostic reasoning system working in a tertiary rare-disease referral "
        "clinic, where patients have typically already been ruled out for the most common "
        "diagnoses elsewhere. Given the clinical findings below, provide your top 3 most "
        "likely diagnoses from this list:\n\n"
        f"{_base_body(disease_list, positive, negative)}\n\n{OUTPUT_INSTRUCTIONS}"
    )


def counterfactual_primary_care(disease_list: str, positive: str, negative: str) -> str:
    """Reframe the base-rate context: primary care, common things are common."""
    return (
        "You are a diagnostic reasoning system working in a general primary care clinic, "
        "where common conditions are far more likely than rare ones. Given the clinical "
        "findings below, provide your top 3 most likely diagnoses from this list:\n\n"
        f"{_base_body(disease_list, positive, negative)}\n\n{OUTPUT_INSTRUCTIONS}"
    )


CONDITIONS = {
    "baseline": baseline,
    "debias": debias,
    "cot": chain_of_thought,
    "counterfactual_rare_clinic": counterfactual_rare_clinic,
    "counterfactual_primary_care": counterfactual_primary_care,
}

# CoT needs room to reason AND still reach the structured answer; other
# conditions occasionally ramble too (observed on Qwen2.5-3B) but converge
# much sooner. A flat low budget silently truncates CoT before it ever
# emits "1./2./3." -- that's a formatting issue, not a model failure, so give
# it a much larger budget.
MAX_TOKENS_BY_CONDITION = {
    "baseline": 300,
    "debias": 400,
    "cot": 700,
    "counterfactual_rare_clinic": 300,
    "counterfactual_primary_care": 300,
}
