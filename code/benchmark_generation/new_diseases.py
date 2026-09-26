"""Curated additions to the DiagRare ontology.

Each entry uses the same schema as the original ontology_diseases.csv
(required/supporting/excluding findings as ';'-joined tokens, classic
textbook associations only -- this is a synthetic reasoning benchmark,
not clinical guidance).

`split` marks the intended role of the disease:
  - "expanded_indist": new disease in one of the ORIGINAL 6 organ systems
    (Cardio, Pulmo, Neuro, Metabolic, Immuno, Hemato). Usable for both
    fine-tuning and evaluation.
  - "expanded_heldout": new disease in a NEW organ system (Renal, GI,
    Dermato) never seen in the original benchmark. Held out from
    fine-tuning; used only to test generalization to unseen organ systems.
"""

NEW_DISEASES = [
    # ---- expanded_indist: new diseases within the original 6 organ systems ----
    dict(disease="Infective_Endocarditis", organ_system="Cardio", prevalence=0.0007, is_rare=True,
         required="fever_persistent;new_heart_murmur;positive_blood_cultures",
         supporting="Osler_nodes;Janeway_lesions;splinter_hemorrhages",
         excluding="no_murmur;negative_blood_cultures", split="expanded_indist"),
    dict(disease="Pneumothorax", organ_system="Pulmo", prevalence=0.001, is_rare=True,
         required="sudden_pleuritic_chest_pain;decreased_breath_sounds_unilateral;dyspnea_sudden_onset",
         supporting="hyperresonance_to_percussion;tracheal_deviation;tall_thin_body_habitus",
         excluding="bilateral_breath_sounds_equal;gradual_onset_weeks", split="expanded_indist"),
    dict(disease="Trigeminal_Neuralgia", organ_system="Neuro", prevalence=0.0002, is_rare=True,
         required="paroxysmal_facial_pain;trigger_zone_present;unilateral_facial_distribution",
         supporting="brief_episodes_seconds;normal_neuro_exam_between_episodes;triggered_by_light_touch",
         excluding="continuous_constant_pain;bilateral_symmetric_pain", split="expanded_indist"),
    dict(disease="Diabetic_Ketoacidosis", organ_system="Metabolic", prevalence=0.008, is_rare=False,
         required="hyperglycemia_severe;ketonemia;metabolic_acidosis",
         supporting="kussmaul_breathing;fruity_breath_odor;polyuria_polydipsia_acute",
         excluding="normal_blood_glucose;absent_ketones", split="expanded_indist"),
    dict(disease="Anaphylaxis", organ_system="Immuno", prevalence=0.003, is_rare=False,
         required="acute_onset_after_exposure;airway_compromise_or_hypotension;urticaria_acute",
         supporting="angioedema;wheezing_acute;recent_allergen_exposure",
         excluding="gradual_onset_over_days;no_exposure_history", split="expanded_indist"),
    dict(disease="Disseminated_Intravascular_Coagulation", organ_system="Hemato", prevalence=0.0004, is_rare=True,
         required="concurrent_bleeding_and_clotting;elevated_D_dimer;low_fibrinogen",
         supporting="thrombocytopenia_acute;schistocytes_on_smear;underlying_sepsis_or_trauma",
         excluding="isolated_bleeding_only;normal_coagulation_panel", split="expanded_indist"),

    # ---- expanded_heldout: Renal (new organ system) ----
    dict(disease="Acute_Kidney_Injury", organ_system="Renal", prevalence=0.02, is_rare=False,
         required="oliguria;rising_creatinine;fluid_overload",
         supporting="hyperkalemia;metabolic_acidosis_renal;uremic_symptoms",
         excluding="normal_creatinine;preserved_urine_output", split="expanded_heldout"),
    dict(disease="Nephrotic_Syndrome", organ_system="Renal", prevalence=0.004, is_rare=False,
         required="proteinuria_heavy;hypoalbuminemia;peripheral_edema_generalized",
         supporting="hyperlipidemia;frothy_urine;rapid_weight_gain",
         excluding="gross_hematuria;normal_serum_albumin", split="expanded_heldout"),
    dict(disease="Polycystic_Kidney_Disease", organ_system="Renal", prevalence=0.0008, is_rare=True,
         required="bilateral_flank_pain;family_history_renal_disease;palpable_bilateral_kidneys",
         supporting="microscopic_hematuria;hypertension_early_onset;renal_cysts_on_imaging",
         excluding="normal_renal_imaging;no_family_history_renal", split="expanded_heldout"),
    dict(disease="Renal_Cell_Carcinoma", organ_system="Renal", prevalence=0.0003, is_rare=True,
         required="flank_mass_palpable;gross_hematuria;unintentional_weight_loss",
         supporting="low_grade_fever;flank_pain_dull;paraneoplastic_polycythemia",
         excluding="no_renal_mass_on_imaging;normal_renal_imaging", split="expanded_heldout"),
    dict(disease="Glomerulonephritis", organ_system="Renal", prevalence=0.002, is_rare=False,
         required="hematuria_with_red_cell_casts;hypertension_new_onset;periorbital_edema",
         supporting="recent_streptococcal_infection;low_serum_complement;proteinuria_mild",
         excluding="normal_urinalysis;no_red_cell_casts", split="expanded_heldout"),
    dict(disease="Nephrolithiasis", organ_system="Renal", prevalence=0.05, is_rare=False,
         required="colicky_flank_pain;microscopic_or_gross_hematuria;pain_radiating_to_groin",
         supporting="nausea_with_pain;dysuria;stone_visible_on_imaging",
         excluding="high_fever_with_sepsis;peritoneal_signs", split="expanded_heldout"),

    # ---- expanded_heldout: GI (new organ system) ----
    dict(disease="Acute_Appendicitis", organ_system="GI", prevalence=0.07, is_rare=False,
         required="RLQ_abdominal_pain;rebound_tenderness;low_grade_fever",
         supporting="anorexia;nausea_with_pain;migratory_periumbilical_to_RLQ_pain",
         excluding="LLQ_abdominal_pain;chronic_symptoms_over_years", split="expanded_heldout"),
    dict(disease="Peptic_Ulcer_Disease", organ_system="GI", prevalence=0.1, is_rare=False,
         required="epigastric_pain;NSAID_use_or_H_pylori_infection;nocturnal_epigastric_pain",
         supporting="bloating;early_satiety;melena_intermittent",
         excluding="RLQ_abdominal_pain;jaundice", split="expanded_heldout"),
    dict(disease="Ulcerative_Colitis", organ_system="GI", prevalence=0.002, is_rare=True,
         required="bloody_diarrhea_chronic;tenesmus;continuous_colonic_involvement",
         supporting="gradual_weight_loss;abdominal_cramping;extraintestinal_arthritis",
         excluding="skip_lesions_on_colonoscopy;perianal_fistula", split="expanded_heldout"),
    dict(disease="Crohn_Disease", organ_system="GI", prevalence=0.0015, is_rare=True,
         required="skip_lesions_on_colonoscopy;perianal_fistula;transmural_inflammation",
         supporting="abdominal_pain_chronic;gradual_weight_loss;nonbloody_diarrhea",
         excluding="continuous_colonic_involvement;bloody_diarrhea_massive", split="expanded_heldout"),
    dict(disease="Acute_Pancreatitis", organ_system="GI", prevalence=0.004, is_rare=False,
         required="epigastric_pain_radiating_to_back;elevated_lipase;alcohol_or_gallstone_history",
         supporting="nausea_with_pain;vomiting_persistent;Grey_Turner_sign",
         excluding="normal_lipase;RLQ_abdominal_pain", split="expanded_heldout"),
    dict(disease="Acute_Cholecystitis", organ_system="GI", prevalence=0.003, is_rare=False,
         required="RUQ_abdominal_pain;positive_Murphy_sign;low_grade_fever",
         supporting="nausea_with_pain;fatty_food_intolerance;gallstones_on_imaging",
         excluding="LUQ_abdominal_pain;normal_abdominal_imaging", split="expanded_heldout"),

    # ---- expanded_heldout: Dermato (new organ system) ----
    dict(disease="Psoriasis", organ_system="Dermato", prevalence=0.03, is_rare=False,
         required="silvery_scaling_plaques;extensor_surface_distribution;nail_pitting",
         supporting="family_history_skin_disease;koebner_phenomenon;psoriatic_joint_pain",
         excluding="pruritus_without_scaling;vesicular_rash", split="expanded_heldout"),
    dict(disease="Cellulitis", organ_system="Dermato", prevalence=0.02, is_rare=False,
         required="unilateral_erythema;warmth_and_swelling;low_grade_fever",
         supporting="recent_skin_trauma;leukocytosis;localized_tenderness",
         excluding="bilateral_symmetric_rash;no_warmth_on_exam", split="expanded_heldout"),
    dict(disease="Allergic_Contact_Dermatitis", organ_system="Dermato", prevalence=0.015, is_rare=False,
         required="pruritic_rash;recent_allergen_exposure;well_demarcated_rash_borders",
         supporting="vesicles_present;linear_rash_pattern;resolves_after_allergen_removal",
         excluding="systemic_symptoms_present;high_fever", split="expanded_heldout"),
    dict(disease="Herpes_Zoster", organ_system="Dermato", prevalence=0.003, is_rare=False,
         required="dermatomal_vesicular_rash;pain_preceding_rash;unilateral_distribution",
         supporting="prior_varicella_history;risk_of_postherpetic_neuralgia;grouped_vesicles",
         excluding="bilateral_symmetric_rash;painless_rash", split="expanded_heldout"),
    dict(disease="Stevens_Johnson_Syndrome", organ_system="Dermato", prevalence=0.00005, is_rare=True,
         required="mucosal_involvement;skin_detachment;recent_new_medication_exposure",
         supporting="high_fever;target_lesions_atypical;positive_Nikolsky_sign",
         excluding="no_recent_medication_change;localized_rash_only", split="expanded_heldout"),
    dict(disease="Erythema_Multiforme", organ_system="Dermato", prevalence=0.001, is_rare=True,
         required="target_lesions_typical;acral_distribution;recent_HSV_infection",
         supporting="mild_mucosal_involvement;self_limited_course;symmetric_distribution",
         excluding="extensive_skin_detachment;recent_new_medication_exposure", split="expanded_heldout"),
]

for _d in NEW_DISEASES:
    _d["n_required"] = len(_d["required"].split(";"))
    _d["n_supporting"] = len(_d["supporting"].split(";"))
    _d["n_excluding"] = len(_d["excluding"].split(";"))
