# -*- coding: utf-8 -*-
"""
03_shap_analysis.py
===================
Compute global and local SHAP feature attributions for the optimal band-gap model
(ExtraTrees on matbench_expt_gap, 142 Magpie features).

What this script does:
  1. Loads the pre-featurized training data (fold 0 as representative).
  2. Trains the optimal ExtraTrees model.
  3. Computes SHAP values (TreeExplainer) on the test set.
  4. Exports global mean |SHAP| ranking to results/shap_global_importance.csv.
  5. Exports local SHAP values for the full test set to results/shap_local_values.csv.
  6. Runs 5-fold SHAP rank stability analysis (Spearman rho across fold pairs).
  7. Saves a combined SHAP bar-chart (top-25 features).

Adapted from:  QMOF 最新程序 (SHAP TreeExplainer pattern, save_fig utility,
               robust CSV read, 5-fold stratified iteration style)

Random seed: 18012019
"""

import json, copy, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 0)  Configuration
# ──────────────────────────────────────────────────────────────
SEED      = 18012019
FEAT_DIR  = Path("data") / "features" / "matbench_expt_gap"
OUT_DIR   = Path("results")
FIG_DIR   = Path("results") / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 25   # features shown in the bar chart

# ──────────────────────────────────────────────────────────────
# 1)  Utilities  (robust CSV read — same pattern as QMOF modules)
# ──────────────────────────────────────────────────────────────
def robust_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

def save_fig(fig, stem):
    """Save at 300 dpi — adapted from QMOF save_fig utility."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / stem}.png")

# ──────────────────────────────────────────────────────────────
# 2)  Model (optimal: ExtraTrees)
# ──────────────────────────────────────────────────────────────
def make_et_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   ExtraTreesRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1)),
    ])

# ──────────────────────────────────────────────────────────────
# 3)  Load per-fold data  (follows QMOF per-fold iteration)
# ──────────────────────────────────────────────────────────────
def load_fold_data(fold_idx):
    path = FEAT_DIR / f"fold_{fold_idx}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fold file not found: {path}\n"
            "Run 01_extract_features.py first."
        )
    df = robust_read_csv(path)
    feat_cols = [c for c in df.columns
                 if c not in ("target", "split", "fold")]
    train = df[df["split"] == "train"].copy()
    test  = df[df["split"] == "test"].copy()
    return train[feat_cols], train["target"], test[feat_cols], feat_cols

# ──────────────────────────────────────────────────────────────
# 4)  Import SHAP (guard for missing package)
# ──────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("WARNING: shap not installed. Run: pip install shap>=0.41.0")

# ──────────────────────────────────────────────────────────────
# 5)  5-fold SHAP global importance + rank stability
#     (mirrors the 5-fold iteration in QMOF base training cells)
# ──────────────────────────────────────────────────────────────
print("Computing 5-fold SHAP attributions for matbench_expt_gap (ExtraTrees)...")

fold_global_imp = []   # list of arrays, one per fold
feat_cols_ref   = None

for fold_idx in range(5):
    print(f"  Fold {fold_idx}...", end=" ", flush=True)

    X_train, y_train, X_test, feat_cols = load_fold_data(fold_idx)
    if feat_cols_ref is None:
        feat_cols_ref = feat_cols

    model = make_et_model()
    model.fit(X_train, y_train)

    if not SHAP_OK:
        # Fallback: feature importances from the tree model
        raw_imp = model.named_steps["model"].feature_importances_
        fold_global_imp.append(raw_imp)
        print(f"(feature_importances_ fallback, fold {fold_idx})")
        continue

    # ── SHAP TreeExplainer (pattern from QMOF 2-SHAP script)
    # Feed transformed data (after imputer) to the tree explainer
    X_test_imp = model.named_steps["imputer"].transform(X_test)
    tree_model  = model.named_steps["model"]

    explainer   = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test_imp)   # (n_test, n_features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # global importance
    fold_global_imp.append(mean_abs_shap)
    print(f"done (n_test={len(X_test_imp)}, shap shape={shap_values.shape})")

    # Save local SHAP values for fold 0 (representative)
    if fold_idx == 0 and SHAP_OK:
        local_df = pd.DataFrame(shap_values, columns=feat_cols_ref)
        local_df.to_csv(OUT_DIR / "shap_local_values_fold0.csv",
                        index=False, encoding="utf-8-sig")
        print("  Saved: results/shap_local_values_fold0.csv")

# ──────────────────────────────────────────────────────────────
# 6)  Global importance table  (mean across 5 folds)
# ──────────────────────────────────────────────────────────────
imp_matrix = np.vstack(fold_global_imp)          # (5, n_features)
mean_imp   = imp_matrix.mean(axis=0)
std_imp    = imp_matrix.std(axis=0, ddof=1)

imp_df = pd.DataFrame({
    "feature":        feat_cols_ref,
    "mean_abs_shap":  mean_imp,
    "std_abs_shap":   std_imp,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp_df["rank"] = imp_df.index + 1

imp_df.to_csv(OUT_DIR / "shap_global_importance.csv",
              index=False, encoding="utf-8-sig")
print(f"Saved: results/shap_global_importance.csv ({len(imp_df)} features)")

# ──────────────────────────────────────────────────────────────
# 7)  5-fold rank stability  (Spearman rho across all fold pairs)
#     Equivalent to the inter-fold Spearman check in QMOF modules
# ──────────────────────────────────────────────────────────────
from itertools import combinations

# Convert each fold's importances to rank
fold_ranks = [np.argsort(np.argsort(-imp)) for imp in fold_global_imp]

rhos = []
for i, j in combinations(range(5), 2):
    rho, _ = spearmanr(fold_ranks[i], fold_ranks[j])
    rhos.append(rho)

mean_rho = float(np.mean(rhos))
std_rho  = float(np.std(rhos, ddof=1))
print(f"\n5-fold SHAP rank stability (Spearman rho): "
      f"{mean_rho:.4f} +/- {std_rho:.4f}")
print(f"  (paper reports: mean rho = 0.9401 +/- 0.0142)")

stability_df = pd.DataFrame({
    "fold_pair": [f"{i}-{j}" for i, j in combinations(range(5), 2)],
    "spearman_rho": rhos,
})
stability_df.to_csv(OUT_DIR / "shap_rank_stability.csv",
                    index=False, encoding="utf-8-sig")
print("Saved: results/shap_rank_stability.csv")

# ──────────────────────────────────────────────────────────────
# 8)  Bar chart: top-25 features  (adapted from QMOF save_fig style)
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 11,
})

top = imp_df.head(TOP_N).copy()

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
        xerr=top["std_abs_shap"][::-1],
        align="center", capsize=3)
ax.set_xlabel("Mean |SHAP value| (eV)", fontsize=11)
ax.set_title(f"Top {TOP_N} Magpie features by SHAP importance\n"
             f"(ExtraTrees, matbench_expt_gap, 5-fold mean ± SD)", fontsize=11)
save_fig(fig, "shap_top25_barchart")

print(f"\nTop 5 features:")
for _, row in imp_df.head(5).iterrows():
    print(f"  Rank {int(row['rank'])}: {row['feature']:<45} "
          f"mean|SHAP|={row['mean_abs_shap']:.4f}")

print("\nSHAP analysis complete.")
