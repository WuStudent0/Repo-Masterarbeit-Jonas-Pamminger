"""
04_regression.py
----------------
Panel fixed-effects regressions testing H1 and H2.

Input:  data/processed/panel_clean.parquet
Output: output/tables/regression_results.csv

Models
------
(1) Baseline:    ROA ~ DOI + controls                      (firm + year FE)
(2) H1 test:     ROA ~ DOI + DOI² + controls               (inverted U-shape)
(3) H2 test:     ROA ~ DOI + DOI² + R&D + DOI×R&D + controls (moderation)

Estimator: linearmodels PanelOLS with two-way fixed effects and firm-clustered SEs.

Reading results
---------------
H1 supported if: β(DOI) > 0  AND  β(DOI²) < 0
H2 supported if: β(DOI × R&D) > 0
Stars: *** p<0.01, ** p<0.05, * p<0.10
"""

import warnings
from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/processed/panel_clean.parquet")
TABLE_PATH = Path("output/tables")
TABLE_PATH.mkdir(parents=True, exist_ok=True)

# ── Load & Set Panel Index ────────────────────────────────────────────────────
# linearmodels requires a MultiIndex: (entity, time)
df = pd.read_parquet(DATA_PATH)
for c in df.select_dtypes(include=["Float64"]).columns:
    df[c] = df[c].astype("float64")
df["doi_x_size"] = df["doi"] * df["ln_at"]
df.to_parquet(Path("data/processed/panel_with_vars.parquet"), index=False)
df = df.set_index(["gvkey", "fyear"])

print(f"Panel: {len(df):,} obs | {df.index.get_level_values('gvkey').nunique():,} firms")

CONTROLS = ["ln_at", "leverage", "capex_intensity", "rd_intensity"]


# ── Helper: two-way FE regression ────────────────────────────────────────────
def run_fe(dep: str, indep: list[str]) -> object:
    """
    Estimate two-way (firm + year) fixed effects with firm-clustered SEs.

    Parameters
    ----------
    dep   : dependent variable name
    indep : list of independent variable names (controls added automatically)
    """
    formula_vars = indep + CONTROLS
    sub = df[[dep, *formula_vars]].dropna()
    formula = f"{dep} ~ {' + '.join(formula_vars)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(formula, data=sub)
    return mod.fit(cov_type="clustered", cluster_entity=True)


# ── Estimate three models ─────────────────────────────────────────────────────
print("\nEstimating models...")
res1 = run_fe("roa", ["doi"])
print("  Model 1 (baseline TWFE) done")

res2 = run_fe("roa", ["doi", "doi_x_size"])
print("  Model 2 (H2: size moderation) done")


# ── Build Results Table ───────────────────────────────────────────────────────
KEY_VARS = ["doi", "doi_x_size"] + CONTROLS

model_labels = ["(1) TWFE Baseline", "(2) TWFE + H2 Moderation"]
models = [res1, res2]

rows = []
for label, res in zip(model_labels, models):
    col: dict = {"Model": label}
    for var in KEY_VARS:
        if var in res.params.index:
            coef  = res.params[var]
            se    = res.std_errors[var]
            pval  = res.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            col[var]          = f"{coef:.3f}{stars}"
            col[f"{var}_se"]  = f"({se:.3f})"
        else:
            col[var]         = ""
            col[f"{var}_se"] = ""
    col["N"]  = f"{int(res.nobs):,}"
    col["R²"] = f"{res.rsquared:.3f}"
    rows.append(col)

results_df = pd.DataFrame(rows).set_index("Model").T
print("\n=== Regression Results ===")
print(results_df.to_string())
results_df.to_csv(TABLE_PATH / "regression_results.csv")
print(f"\nSaved regression_results.csv")


# ── H1 Diagnostic ─────────────────────────────────────────────────────────────
print("\n--- H1 Diagnostic ---")
b_doi = res1.params["doi"]
p_doi = res1.pvalues["doi"]
stars_h1 = "***" if p_doi < 0.01 else "**" if p_doi < 0.05 else "*" if p_doi < 0.1 else "(n.s.)"
print(f"  beta(DOI) = {b_doi:.4f} {stars_h1}  (p = {p_doi:.4f})")
if b_doi < 0 and p_doi < 0.05:
    print("  H1 supported: DOI negatively affects RoA")
else:
    print("  H1 not supported at conventional significance levels")

# ── H2 Diagnostic ─────────────────────────────────────────────────────────────
print("\n--- H2 Diagnostic ---")
if "doi_x_size" in res2.params.index:
    b_mod = res2.params["doi_x_size"]
    p_mod = res2.pvalues["doi_x_size"]
    stars_h2 = "***" if p_mod < 0.01 else "**" if p_mod < 0.05 else "*" if p_mod < 0.1 else "(n.s.)"
    print(f"  beta(DOI x Size) = {b_mod:.4f} {stars_h2}  (p = {p_mod:.4f})")
    if b_mod > 0 and p_mod < 0.1:
        print("  H2 supported: firm size positively moderates DOI-performance")
    else:
        print("  H2 not supported at conventional significance levels")

print("""
─────────────────────────────────────────────────────────────
Interpretation guide:
  Stars: *** p<0.01, ** p<0.05, * p<0.10
  SEs in parentheses, clustered at firm level
  All models: firm FE + year FE (two-way fixed effects)
─────────────────────────────────────────────────────────────
""")
