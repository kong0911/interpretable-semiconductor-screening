# Interpretable Machine Learning for Candidate Semiconductor Prioritization

**Paper title:** Selection of Candidate Semiconductor Materials with Target Band Gaps Based on Interpretable Machine Learning and Multi-Criteria Prioritization

**Journal:** Journal of Electronic Materials (JEMS) – Revised Submission

**DOI (Zenodo):** 10.5281/zenodo.10892341 *(placeholder; replace with live DOI after upload)*

---

## Repository Contents

```
GitHub/
├── README.md                          This file
├── data/
│   ├── dataset_identifiers.json       Matbench task keys and versions
│   ├── candidate_pool_info.json       Materials Project candidate pool metadata
│   └── overlap_removal_list.csv       18 overlapping compositions removed from candidate pool
├── code/
│   ├── 01_extract_features.py         Magpie descriptor extraction (matminer)
│   ├── 02_train_evaluate_5fold.py     Matbench 5-fold CV training and evaluation
│   ├── 03_shap_analysis.py            Global + local SHAP attribution
│   ├── 04_candidate_ranking.py        Multi-criteria candidate prioritization
│   └── environment.yml                Conda environment specification
├── results/
│   ├── fold_by_fold_cv_results.json   Authoritative 5-fold CV metrics per model/fold
│   ├── candidate_list_full.csv        Complete list of 3,258 deduplicated candidates
│   │                                  (MP-ID, formula, predicted Eg, ref Eg, ΔEg, priority)
│   └── shap_top25_features.csv        Top 25 SHAP features with mean |SHAP| per fold
└── supplementary_tables/
    ├── Table_S4_hyperparameters.csv
    ├── Table_S5_threshold_sensitivity.csv
    ├── Table_S6_shap_feature_decomposition.csv
    ├── Table_S7_composition_overlap.csv
    ├── Table_S8_fold_by_fold_cv_results.csv
    └── Table_V_literature_cross_validation.csv
```

---

## Quick Start

```bash
# 1. Create conda environment
conda env create -f code/environment.yml
conda activate mat-ml-gap

# 2. Extract features (downloads Matbench datasets automatically)
python code/01_extract_features.py

# 3. Train and evaluate (5-fold CV, all tasks)
python code/02_train_evaluate_5fold.py

# 4. Run SHAP analysis
python code/03_shap_analysis.py

# 5. Prioritize candidates
python code/04_candidate_ranking.py
```

---

## Key Numerical Results (authoritative — from fold_by_fold_cv_results.json)

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| matbench_expt_gap | ExtraTrees (Optimal) | MAE (5-fold mean ± SD) | 0.3952 ± 0.0187 eV |
| matbench_expt_gap | RandomForest | MAE (5-fold mean ± SD) | 0.4417 ± 0.0181 eV |
| matbench_expt_is_metal | HistGradientBoosting (Optimal) | ROC-AUC (5-fold) | 0.9761 ± 0.0024 |
| matbench_glass | RandomForest (Optimal) | ROC-AUC (5-fold) | 0.9500 ± 0.0095 |

---

## Candidate Priority Tiers (after overlap exclusion, N = 3,258)

| Tier | Criterion | Count |
|------|-----------|-------|
| High Priority (Tier 1) | ΔEg ≤ 0.30 eV, Eg(ref) > 0 eV | **486** |
| Medium Priority (Tier 2) | 0.30 < ΔEg ≤ 0.90 eV, Eg(ref) > 0 eV | **890** |
| Verification Required (Tier 3) | ΔEg > 0.90 eV or Eg(ref) = 0 eV | **1,882** |
| Low / Excluded (Tier 4) | Outside window, metallic, or glass-forming | — |

Random seed throughout: `random_state = 18012019`

---

## Citation

> [Authors]. Selection of Candidate Semiconductor Materials with Target Band Gaps Based on
> Interpretable Machine Learning and Multi-Criteria Prioritization.
> *Journal of Electronic Materials*, 2024. DOI: [pending]

---

## License

Data: CC-BY 4.0. Code: MIT License.
