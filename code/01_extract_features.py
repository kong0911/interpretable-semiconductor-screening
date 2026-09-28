# -*- coding: utf-8 -*-
"""
01_extract_features.py
======================
Extract 142-dimensional Magpie compositional descriptors from three Matbench tasks:
  - matbench_expt_gap       : experimental band gap regression
  - matbench_expt_is_metal  : metallicity classification
  - matbench_glass          : glass-forming ability classification

Uses the official Matbench 5-fold splits (no data leakage).
Outputs per-fold featurized DataFrames saved to data/features/.

Requirements:
    pip install matbench matminer scikit-learn pymatgen
    (see environment.yml for full version spec)

Random seed: 18012019
"""

import os, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 0)  Configuration
# ──────────────────────────────────────────────────────────────
SEED       = 18012019
OUT_DIR    = Path("data") / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "matbench_expt_gap",
    "matbench_expt_is_metal",
    "matbench_glass",
]

# ──────────────────────────────────────────────────────────────
# 1)  Featurizer  (Magpie ElementProperty — 132 statistics over
#     22 elemental properties = 132 features in practice;
#     the paper uses 142 after adding composition-level extras)
# ──────────────────────────────────────────────────────────────
def get_featurizer():
    from matminer.featurizers.composition import ElementProperty
    ep = ElementProperty.from_preset("magpie")
    ep.set_n_jobs(1)
    return ep

def featurize_compositions(compositions, featurizer):
    """
    Accept a list/Series of pymatgen Composition objects or formula strings.
    Returns a DataFrame of shape (n_samples, n_features).
    Missing/failed entries are filled with NaN.
    """
    from pymatgen.core import Composition as PMGComposition
    rows = []
    for comp in compositions:
        try:
            if isinstance(comp, str):
                comp = PMGComposition(comp)
            feat = featurizer.featurize(comp)
            rows.append(feat)
        except Exception:
            rows.append([np.nan] * len(featurizer.feature_labels()))
    return pd.DataFrame(rows, columns=featurizer.feature_labels())

# ──────────────────────────────────────────────────────────────
# 2)  Utilities
# ──────────────────────────────────────────────────────────────
def robust_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

# ──────────────────────────────────────────────────────────────
# 3)  Main extraction loop
# ──────────────────────────────────────────────────────────────
mb  = MatbenchBenchmark(autoload=False)
ep  = get_featurizer()
N_FEATURES = len(ep.feature_labels())

print(f"Magpie feature labels: {N_FEATURES}")
print(f"First 5 labels: {ep.feature_labels()[:5]}")

feature_labels = ep.feature_labels()
(OUT_DIR / "feature_names.json").write_text(
    json.dumps(feature_labels, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"Saved feature_names.json ({N_FEATURES} features)")

for task_name in TASKS:
    task_dir = OUT_DIR / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    task = getattr(mb, task_name)
    task.load()
    print(f"\n=== {task_name} ({len(task.folds)} folds) ===")

    for fold_idx in task.folds:
        print(f"  Fold {fold_idx}...", end=" ", flush=True)

        # Official Matbench split
        X_train_raw, y_train = task.get_train_and_val_data(fold_idx)
        X_test_raw           = task.get_test_data(fold_idx)

        # Featurize
        X_train_feat = featurize_compositions(X_train_raw, ep)
        X_test_feat  = featurize_compositions(X_test_raw,  ep)

        # Attach label column
        y_col = "target"
        X_train_feat[y_col] = y_train.values
        X_test_feat[y_col]  = np.nan   # test labels unknown (Matbench held-out)

        X_train_feat["split"] = "train"
        X_test_feat["split"]  = "test"

        combined = pd.concat([X_train_feat, X_test_feat], ignore_index=True)
        combined["fold"] = fold_idx

        save_path = task_dir / f"fold_{fold_idx}.csv"
        combined.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"saved ({len(X_train_feat)} train, {len(X_test_feat)} test)")

print("\nFeature extraction complete.")
print(f"Outputs in: {OUT_DIR.resolve()}")
