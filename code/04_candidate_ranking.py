# -*- coding: utf-8 -*-
"""
04_candidate_ranking.py
=======================
Four-tier multi-criteria candidate prioritization for photovoltaic-oriented
semiconductor band-gap screening.

Pipeline:
  1. Load candidate pool predictions (MP-DFT band gap as reference).
  2. Apply target-window filter: Eg_pred in [1.10, 1.80] eV.
  3. Apply metallicity filter: P(metal) < 0.20  (HistGradientBoosting classifier).
  4. Apply glass-forming filter: P(glass) < 0.50 (ExtraTrees classifier).
  5. Compute |delta_Eg| = |Eg_pred - Eg_ref| for candidates with Eg_ref > 0.
  6. Assign four-tier priority:
       Tier 1 (High Priority)         : delta_Eg <= 0.30 eV  AND Eg_ref > 0
       Tier 2 (Medium Priority)       : 0.30 < delta_Eg <= 0.90 AND Eg_ref > 0
       Tier 3 (Verification Required) : delta_Eg > 0.90  OR  Eg_ref = 0
       Tier 4 (Low/Excluded)          : outside window or filtered by step 3-4
  7. Remove composition-level duplicates (overlap with Matbench training set).
  8. Export complete candidate list with priority labels.

Adapted from QMOF candidate-screening logic (模块8, two-stage window + constraint).

Final counts (authoritative, see paper Table II and candidate_pool_info.json):
  Tier 1 High Priority           :   486 candidates
  Tier 2 Medium Priority         :   890 candidates  (after overlap exclusion)
  Tier 3 Verification Required   : 1882 candidates  (after overlap exclusion)
  Total after dedup               : 3258 candidates  (raw: 3263)

Random seed: 18012019
"""

import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, ExtraTreesClassifier,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 0)  Configuration  (match paper thresholds exactly)
# ──────────────────────────────────────────────────────────────
SEED             = 18012019
TARGET_WIN_MIN   = 1.10    # eV
TARGET_WIN_MAX   = 1.80    # eV
P_METAL_THRESH   = 0.20    # P(metal) >= threshold -> excluded
P_GLASS_THRESH   = 0.50    # P(glass) >= threshold -> excluded
T1_HIGH          = 0.30    # delta_Eg <= T1_HIGH  -> High Priority
T2_MEDIUM        = 0.90    # delta_Eg <= T2_MEDIUM -> Medium Priority

FEAT_DIR  = Path("data") / "features"
OUT_DIR   = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1)  Utilities  (robust CSV read — same as QMOF read_csv_robust)
# ──────────────────────────────────────────────────────────────
def robust_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

# ──────────────────────────────────────────────────────────────
# 2)  Priority assignment function
#     (four-tier rule — adapted from QMOF in_window_point / mask logic)
# ──────────────────────────────────────────────────────────────
def assign_priority(row):
    """
    Assign a four-tier priority label and integer tier to a candidate row.
    Columns expected: Eg_pred, Eg_ref, P_metal, P_glass, delta_Eg.
    Returns (tier: int, label: str).
    """
    Eg_pred = float(row.get("Eg_pred_eV", np.nan))
    Eg_ref  = float(row.get("Eg_ref_eV",  0.0))
    P_metal = float(row.get("P_metal",    1.0))
    P_glass = float(row.get("P_glass",    1.0))

    # Tier 4: outside target window
    if not (TARGET_WIN_MIN <= Eg_pred <= TARGET_WIN_MAX):
        return 4, "Low Priority / Excluded (outside window)"

    # Tier 4: predicted metallic
    if P_metal >= P_METAL_THRESH:
        return 4, "Low Priority / Excluded (metallic)"

    # Tier 4: glass-forming
    if P_glass >= P_GLASS_THRESH:
        return 4, "Low Priority / Excluded (glass-forming)"

    # Compute delta_Eg
    if Eg_ref > 0:
        delta = abs(Eg_pred - Eg_ref)
    else:
        delta = np.inf

    # Tier 1
    if Eg_ref > 0 and delta <= T1_HIGH:
        return 1, "High Priority"

    # Tier 2
    if Eg_ref > 0 and delta <= T2_MEDIUM:
        return 2, "Medium Priority"

    # Tier 3: large discrepancy or no reference
    return 3, "Verification Required"

# ──────────────────────────────────────────────────────────────
# 3)  Load or build candidate pool
#     In a full run, candidates come from Matbench MP gap predictions.
#     Here we load from results/candidate_list_full.csv if it exists,
#     or build a minimal demo from the provided data files.
# ──────────────────────────────────────────────────────────────
cand_path = OUT_DIR / "candidate_list_full.csv"

if cand_path.exists():
    print(f"Loading candidate pool from: {cand_path}")
    df = robust_read_csv(cand_path)
    # Strip comment lines
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")].copy()
else:
    # ── Build demo pool (illustrative; replace with real MP predictions)
    print("candidate_list_full.csv not found — building illustrative demo pool.")
    print("In full reproduction, this file is generated by running Steps 1-2")
    print("on the Materials Project candidate compositions.")

    rng = np.random.RandomState(SEED)
    N_DEMO = 200
    df = pd.DataFrame({
        "mp_id":      [f"mp-{i:05d}" for i in rng.randint(10000, 99999, N_DEMO)],
        "formula":    [f"DemoComp{i}" for i in range(N_DEMO)],
        "Eg_pred_eV": rng.uniform(0.8, 2.2, N_DEMO),
        "Eg_ref_eV":  np.where(rng.rand(N_DEMO) > 0.3,
                               rng.uniform(0.0, 2.5, N_DEMO), 0.0),
        "P_metal":    rng.uniform(0.0, 0.5, N_DEMO),
        "P_glass":    rng.uniform(0.0, 0.3, N_DEMO),
    })

# ──────────────────────────────────────────────────────────────
# 4)  Compute delta_Eg
# ──────────────────────────────────────────────────────────────
df["Eg_pred_eV"] = pd.to_numeric(df["Eg_pred_eV"], errors="coerce")
df["Eg_ref_eV"]  = pd.to_numeric(df.get("Eg_ref_eV", 0), errors="coerce").fillna(0.0)
df["P_metal"]    = pd.to_numeric(df.get("P_metal", 0.0), errors="coerce").fillna(0.0)
df["P_glass"]    = pd.to_numeric(df.get("P_glass", 0.0), errors="coerce").fillna(0.0)

df["delta_Eg_eV"] = np.where(
    df["Eg_ref_eV"] > 0,
    (df["Eg_pred_eV"] - df["Eg_ref_eV"]).abs(),
    np.inf
)

# ──────────────────────────────────────────────────────────────
# 5)  Assign priority tiers  (vectorized via apply)
#     Mirrors QMOF candidates_all construction and mask logic
# ──────────────────────────────────────────────────────────────
result = df.apply(assign_priority, axis=1, result_type="expand")
result.columns = ["priority_tier", "priority_label"]
df = pd.concat([df, result], axis=1)

# ──────────────────────────────────────────────────────────────
# 6)  Overlap removal (deduplication)
#     Load the overlap list; mark overlapping compositions for exclusion.
# ──────────────────────────────────────────────────────────────
overlap_path = Path("data") / "overlap_removal_list.csv"
if overlap_path.exists():
    overlap_df = robust_read_csv(overlap_path)
    overlap_formulas = set(
        overlap_df["canonical_reduced_formula"].astype(str).str.strip()
    )
    if "formula" in df.columns:
        from pymatgen.core import Composition as PMGComposition
        def canonical(f):
            try:
                return PMGComposition(f).reduced_formula
            except Exception:
                return str(f).strip()

        df["canonical_formula"] = df["formula"].astype(str).apply(canonical)
        overlap_mask = df["canonical_formula"].isin(overlap_formulas)
        n_removed = int(overlap_mask.sum())
        df = df[~overlap_mask].copy()
        print(f"Overlap removal: {n_removed} entries removed.")
    else:
        print("No 'formula' column found — skipping overlap removal.")
else:
    print(f"Overlap list not found at {overlap_path} — skipping.")

# ──────────────────────────────────────────────────────────────
# 7)  Tier counts
# ──────────────────────────────────────────────────────────────
tier_counts = df["priority_tier"].value_counts().sort_index()
candidates_in_window = df[df["priority_tier"].isin([1, 2, 3])].copy()

print("\n=== Priority Tier Summary ===")
for tier, label in [(1, "High Priority"),
                    (2, "Medium Priority"),
                    (3, "Verification Required"),
                    (4, "Low / Excluded")]:
    n = int((df["priority_tier"] == tier).sum())
    print(f"  Tier {tier} ({label}): {n}")

n_total = int(df["priority_tier"].isin([1, 2, 3]).sum())
print(f"\n  Total in target window (Tiers 1-3): {n_total}")
print(f"  (Paper values: 486 + 890 + 1882 = 3258 after overlap exclusion)")

# ──────────────────────────────────────────────────────────────
# 8)  Export outputs
# ──────────────────────────────────────────────────────────────
# Full candidate list with tiers
df.to_csv(OUT_DIR / "candidate_list_prioritized.csv",
          index=False, encoding="utf-8-sig")

# Tier-1 only (High Priority)
tier1 = df[df["priority_tier"] == 1].copy()
tier1.to_csv(OUT_DIR / "candidates_high_priority.csv",
             index=False, encoding="utf-8-sig")

# Tier-1 and Tier-2 (High + Medium)
tier12 = df[df["priority_tier"].isin([1, 2])].copy()
tier12.to_csv(OUT_DIR / "candidates_high_medium.csv",
              index=False, encoding="utf-8-sig")

# Summary table
summary = pd.DataFrame([
    {"tier": 1, "label": "High Priority",         "criterion": "delta_Eg <= 0.30 eV AND Eg_ref > 0",              "n": int((df["priority_tier"]==1).sum())},
    {"tier": 2, "label": "Medium Priority",        "criterion": "0.30 < delta_Eg <= 0.90 eV AND Eg_ref > 0",       "n": int((df["priority_tier"]==2).sum())},
    {"tier": 3, "label": "Verification Required",  "criterion": "delta_Eg > 0.90 eV OR Eg_ref = 0",                "n": int((df["priority_tier"]==3).sum())},
    {"tier": 4, "label": "Low / Excluded",         "criterion": "Outside [1.10,1.80] eV OR P(metal)>=0.20 OR glass","n": int((df["priority_tier"]==4).sum())},
])
summary.to_csv(OUT_DIR / "priority_tier_summary.csv",
               index=False, encoding="utf-8-sig")

print("\nOutputs saved to results/:")
print("  candidate_list_prioritized.csv  — all candidates with tier labels")
print("  candidates_high_priority.csv    — Tier 1 only")
print("  candidates_high_medium.csv      — Tiers 1+2")
print("  priority_tier_summary.csv       — tier count summary")
print("\nCandidate prioritization complete.")
