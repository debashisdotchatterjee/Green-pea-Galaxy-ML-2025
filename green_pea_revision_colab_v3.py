# ============================================================
# Green Pea Galaxy Revised Analysis Pipeline (v3)
# Colab-ready, robust to sklearn/seaborn version differences
# ============================================================
# Scientific framing:
# This code treats METALLICITY from Cardamone-style catalogues as an
# adopted strong-line/calibration-scale target unless a direct-Te target
# column is explicitly present. Do NOT interpret low RMSE as recovery of
# physically robust oxygen abundance unless direct-Te/photoionization
# abundances are used as target.
# ============================================================

# -------------------------
# 0. User configuration
# -------------------------
DATA_PATH = None  # e.g. "/content/peas_tbl4.csv"; leave None to upload/search in Colab
OUTPUT_DIR = "green_pea_revised_outputs_v3"
RANDOM_SEED = 20260616
N_SPLITS_CV = 5
N_REPEATS_CV = 5           # 5*5 = 25 fold-runs; increase to 20 for final heavy run if desired
RUN_BAYESIAN_MODEL = True  # full-data heteroscedastic Bayesian calibration model
RUN_BAYESIAN_CV = False    # optional; slow. Full-data Bayesian diagnostics are enough for first run.
RUN_OPTIONAL_MLP_SENSITIVITY = False # optional non-Bayesian neural-net sensitivity only; keep False for first stable run

# If True, try passing sample weights to sklearn baselines that explicitly support it.
# False is safer and avoids version-specific sklearn errors. Weighted_RMSE is still computed.
FIT_BASELINES_WITH_SAMPLE_WEIGHT = False
RUN_GAUSSIAN_PROCESS_BASELINE = False  # GPR is optional and slow; keep False for stable first run

# -------------------------
# 1. Imports and setup
# -------------------------
import os
import re
import sys
import json
import math
import time
import shutil
import zipfile
import warnings
import inspect
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    import seaborn as sns
    HAVE_SEABORN = True
    sns.set_theme(style="whitegrid", context="notebook")
except Exception:
    HAVE_SEABORN = False

from scipy import stats

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
import joblib

try:
    from IPython.display import display, Markdown
    HAVE_IPYTHON = True
except Exception:
    HAVE_IPYTHON = False

np.random.seed(RANDOM_SEED)

OUT = Path(OUTPUT_DIR)
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
MODEL_DIR = OUT / "models"
REPORT_DIR = OUT / "reports"
for d in [OUT, FIG_DIR, TAB_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_LINES = []

def log(msg=""):
    msg = str(msg)
    print(msg)
    LOG_LINES.append(msg)


def display_df(df, n=20, title=None):
    if title:
        log(f"\n{title}")
    if HAVE_IPYTHON:
        display(df.head(n) if hasattr(df, "head") else df)
    else:
        print(df.head(n) if hasattr(df, "head") else df)


def save_table(df, name, index=False):
    path = TAB_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    log(f"Saved table: {name} -> {path}")
    return path


def display_and_save_fig(fig, filename, title="figure"):
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    log(f"Saved figure: {title} -> {path}")
    if HAVE_IPYTHON:
        display(fig)
    plt.close(fig)
    return path


def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_weighted_rmse(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))

# -------------------------
# 2. Data upload/loading
# -------------------------

def find_candidate_data_file():
    candidates = []
    search_dirs = [Path.cwd(), Path("/content"), Path("/mnt/data")]
    for sd in search_dirs:
        if not sd.exists():
            continue
        for ext in ["*.csv", "*.CSV", "*.xlsx", "*.xls", "*.txt", "*.dat", "*.tsv"]:
            candidates.extend(sd.glob(ext))
    # Prefer files with pea/table names, then latest modified.
    candidates = sorted(set(candidates), key=lambda p: ("pea" not in p.name.lower(), "tbl" not in p.name.lower(), -p.stat().st_mtime))
    return candidates[0] if candidates else None


def maybe_upload_in_colab():
    try:
        from google.colab import files
        log("Please upload the Green Pea dataset CSV/XLSX/TXT file now.")
        uploaded = files.upload()
        if not uploaded:
            return None
        first_name = list(uploaded.keys())[0]
        return Path("/content") / first_name
    except Exception:
        return None


def read_dataset(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix in [".tsv", ".dat"]:
        return pd.read_csv(path, sep="\t", comment=None)
    if suffix == ".txt":
        # Try comma, tab, whitespace in order.
        for sep in [",", "\t", r"\s+"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
        return pd.read_csv(path)
    return pd.read_csv(path)

if DATA_PATH is not None:
    data_path = Path(DATA_PATH)
else:
    data_path = find_candidate_data_file()
    # In Colab, if no good local candidate, upload.
    if data_path is None or data_path.name.startswith("green_pea_revision"):
        uploaded_path = maybe_upload_in_colab()
        if uploaded_path is not None:
            data_path = uploaded_path

if data_path is None:
    raise FileNotFoundError("No dataset found. Set DATA_PATH or upload peas_tbl4.csv in Colab.")

raw_df = read_dataset(data_path)
log(f"Loaded dataset: {data_path} with shape {raw_df.shape}")

# -------------------------
# 3. Column normalization and numeric coercion
# -------------------------

def normalize_col(c):
    c = str(c).strip()
    c = re.sub(r"^#+", "", c).strip()
    c = c.replace("+", "PLUS")
    c = c.replace("/", "_")
    c = re.sub(r"[^0-9A-Za-z]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c.upper()

original_columns = list(raw_df.columns)
normalized_columns = [normalize_col(c) for c in original_columns]
# Ensure uniqueness.
seen = {}
unique_cols = []
for c in normalized_columns:
    if c not in seen:
        seen[c] = 0
        unique_cols.append(c)
    else:
        seen[c] += 1
        unique_cols.append(f"{c}_{seen[c]}")

column_map = pd.DataFrame({"original_column": original_columns, "normalized_column": unique_cols})
df = raw_df.copy()
df.columns = unique_cols
save_table(column_map, "column_name_mapping")
display_df(column_map, title="Column name mapping")

# Convert object columns to numeric whenever possible; preserve IDs as object if conversion fails.
for col in df.columns:
    if df[col].dtype == object:
        cleaned = df[col].astype(str).str.strip()
        cleaned = cleaned.replace({"": np.nan, "--": np.nan, "...": np.nan, "NAN": np.nan, "nan": np.nan, "None": np.nan})
        converted = pd.to_numeric(cleaned, errors="coerce")
        # If at least 50% are numeric, use numeric.
        if converted.notna().mean() >= 0.50:
            df[col] = converted
        else:
            df[col] = cleaned.replace({"nan": np.nan})

# -------------------------
# 4. Detect target and uncertainty columns
# -------------------------

def first_existing(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# Direct-Te candidates first. Your current file likely does NOT contain these.
direct_target_candidates = [
    "METALLICITY_TE", "TE_METALLICITY", "DIRECT_TE_METALLICITY", "DIRECT_METALLICITY",
    "O_H_TE", "OH_TE", "OXYGEN_ABUNDANCE_TE", "DIRECT_OH", "TE_OH",
    "12PLUSLOGOH_TE", "12_LOG_OH_TE", "T_E_METALLICITY"
]
strong_line_target_candidates = [
    "METALLICITY", "OH", "O_H", "OXYGEN_ABUNDANCE", "12PLUSLOGOH", "12_LOG_OH", "LOGOH12"
]

target_col = first_existing(direct_target_candidates)
target_kind = "direct_Te_or_physical_abundance" if target_col else None
if target_col is None:
    target_col = first_existing(strong_line_target_candidates)
    target_kind = "strong_line_calibration" if target_col else None
if target_col is None:
    raise ValueError(f"Could not detect metallicity target. Available columns: {list(df.columns)}")

df["Y_TARGET"] = pd.to_numeric(df[target_col], errors="coerce")

err_upper_col = first_existing([f"{target_col}_ERR_UPPER", "METALLICITY_ERR_UPPER", "ERR_UPPER", "YERR_UPPER", "OH_ERR_UPPER"])
err_lower_col = first_existing([f"{target_col}_ERR_LOWER", "METALLICITY_ERR_LOWER", "ERR_LOWER", "YERR_LOWER", "OH_ERR_LOWER"])
err_sym_col = first_existing([f"{target_col}_ERR", "METALLICITY_ERR", "ERR", "YERR", "OH_ERR"])

log(f"Detected target column: {target_col} ({target_kind})")
log(f"Detected target error columns: upper={err_upper_col}, lower={err_lower_col}, symmetric={err_sym_col}")

if target_kind == "strong_line_calibration":
    warning_text = """
IMPORTANT SCIENTIFIC WARNING
----------------------------
The detected target appears to be a strong-line/calibration metallicity.
Report results as empirical prediction/reproduction of the adopted metallicity calibration scale.
Do NOT claim direct recovery of physically robust oxygen abundance unless direct-Te or
photoionization-model abundance targets are added and used.
""".strip()
    log("\n" + warning_text + "\n")
    (REPORT_DIR / "scientific_warning.txt").write_text(warning_text)

# Error construction.
if err_lower_col is not None:
    df["YERR_LOW_RAW"] = pd.to_numeric(df[err_lower_col], errors="coerce").abs()
else:
    df["YERR_LOW_RAW"] = np.nan

if err_upper_col is not None:
    df["YERR_UP_RAW"] = pd.to_numeric(df[err_upper_col], errors="coerce").abs()
else:
    df["YERR_UP_RAW"] = np.nan

if err_sym_col is not None:
    df["YERR_SYM_RAW"] = pd.to_numeric(df[err_sym_col], errors="coerce").abs()
else:
    df["YERR_SYM_RAW"] = np.nan

# Fill missing error values conservatively but keep flags.
if df["YERR_LOW_RAW"].notna().any() and df["YERR_UP_RAW"].notna().any():
    df["YERR_SYM"] = df[["YERR_LOW_RAW", "YERR_UP_RAW"]].mean(axis=1)
    df["YERR_LOW"] = df["YERR_LOW_RAW"]
    df["YERR_UP"] = df["YERR_UP_RAW"]
elif df["YERR_SYM_RAW"].notna().any():
    df["YERR_SYM"] = df["YERR_SYM_RAW"]
    df["YERR_LOW"] = df["YERR_SYM_RAW"]
    df["YERR_UP"] = df["YERR_SYM_RAW"]
else:
    df["YERR_SYM"] = np.nan
    df["YERR_LOW"] = np.nan
    df["YERR_UP"] = np.nan

fallback_err = float(np.nanmedian(df["YERR_SYM"])) if np.isfinite(np.nanmedian(df["YERR_SYM"])) else 0.10
fallback_err = max(fallback_err, 0.03)
df["YERR_WAS_MISSING"] = df["YERR_SYM"].isna()
df["YERR_SYM"] = df["YERR_SYM"].fillna(fallback_err).clip(lower=0.03)
df["YERR_LOW"] = df["YERR_LOW"].fillna(df["YERR_SYM"]).clip(lower=0.03)
df["YERR_UP"] = df["YERR_UP"].fillna(df["YERR_SYM"]).clip(lower=0.03)

# -------------------------
# 5. Predictor construction
# -------------------------

def has_col(c):
    return c in df.columns

feature_cols = []

if has_col("REDSHIFT"):
    df["REDSHIFT"] = pd.to_numeric(df["REDSHIFT"], errors="coerce")
    feature_cols.append("REDSHIFT")

if has_col("OIII_EW"):
    df["OIII_EW"] = pd.to_numeric(df["OIII_EW"], errors="coerce")
    df["LOG_OIII_EW"] = np.log10(1.0 + df["OIII_EW"].clip(lower=0))
    feature_cols.append("LOG_OIII_EW")
    log("Created transformed feature LOG_OIII_EW from OIII_EW")

if has_col("SFR"):
    df["SFR"] = pd.to_numeric(df["SFR"], errors="coerce")
    df["LOG_SFR"] = np.log10(1.0 + df["SFR"].clip(lower=0))
    feature_cols.append("LOG_SFR")
    log("Created transformed feature LOG_SFR from SFR")

if has_col("L_FUV"):
    df["L_FUV"] = pd.to_numeric(df["L_FUV"], errors="coerce")
    df["LOG_L_FUV"] = np.log10(1.0 + df["L_FUV"].clip(lower=0))
    feature_cols.append("LOG_L_FUV")
    log("Created transformed feature LOG_L_FUV from L_FUV")

# M_STELLAR in Cardamone table is already log10(M*/Msun). Detect this safely.
if has_col("M_STELLAR"):
    df["M_STELLAR"] = pd.to_numeric(df["M_STELLAR"], errors="coerce")
    med_m = np.nanmedian(df["M_STELLAR"])
    if np.isfinite(med_m) and med_m < 20:
        df["MSTAR_LOG"] = df["M_STELLAR"]
        log("Created/recognized MSTAR_LOG from M_STELLAR (already log-scale)")
    else:
        df["MSTAR_LOG"] = np.log10(df["M_STELLAR"].clip(lower=1e-30))
        log("Created MSTAR_LOG as log10(M_STELLAR)")
    feature_cols.append("MSTAR_LOG")

if not feature_cols:
    raise ValueError("No usable predictors detected. Need columns such as REDSHIFT, OIII_EW, SFR, L_FUV, M_STELLAR.")

# Keep rows with target; predictors may contain missing values, handled by imputation inside CV.
model_df = df.loc[df["Y_TARGET"].notna(), feature_cols + ["Y_TARGET", "YERR_LOW", "YERR_UP", "YERR_SYM", "YERR_WAS_MISSING"]].copy()
# Remove completely empty predictor rows.
model_df = model_df.loc[model_df[feature_cols].notna().any(axis=1)].reset_index(drop=True)

X = model_df[feature_cols].copy()
y = model_df["Y_TARGET"].to_numpy(dtype=float)
yerr_low_arr = model_df["YERR_LOW"].to_numpy(dtype=float)
yerr_up_arr = model_df["YERR_UP"].to_numpy(dtype=float)
yerr_sym_arr = model_df["YERR_SYM"].to_numpy(dtype=float)
N = len(model_df)

log(f"Final modelling sample size: N={N}")
log(f"Final predictors: {feature_cols}")
display_df(model_df.head(15), title="Cleaned modelling data head")
save_table(model_df.head(30), "cleaned_modeling_data_head")

if N < 20:
    raise ValueError(f"Too few complete target rows for reliable repeated CV: N={N}")

# Missingness summary on full normalized df.
miss_summary = pd.DataFrame({
    "column": df.columns,
    "missing_count": [int(df[c].isna().sum()) for c in df.columns],
    "missing_percent": [float(100 * df[c].isna().mean()) for c in df.columns]
}).sort_values(["missing_percent", "missing_count"], ascending=False)
display_df(miss_summary, title="Missingness summary")
save_table(miss_summary, "missingness_summary")

# -------------------------
# 6. Exploratory plots/tables
# -------------------------
# 6.1 Missingness map
fig, ax = plt.subplots(figsize=(12, max(4, 0.22 * len(df.columns))))
miss_matrix = df.isna().T.astype(int).to_numpy()
ax.imshow(miss_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
ax.set_yticks(np.arange(len(df.columns)))
ax.set_yticklabels(df.columns, fontsize=8)
ax.set_xlabel("Row index")
ax.set_title("Missingness map (yellow = missing)")
display_and_save_fig(fig, "01_missingness_map.png", "Missingness map")

# 6.2 Target distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y, bins=min(14, max(6, int(np.sqrt(N)))), alpha=0.75, edgecolor="black")
ax.axvline(np.mean(y), color="black", linestyle="--", label=f"mean={np.mean(y):.3f}")
ax.axvline(np.median(y), color="dimgray", linestyle=":", label=f"median={np.median(y):.3f}")
ax.set_xlabel(r"Target metallicity: $12+\log(\mathrm{O/H})$")
ax.set_ylabel("Count")
ax.set_title("Distribution of adopted metallicity target")
ax.legend()
display_and_save_fig(fig, "02_target_distribution.png", "Target distribution")

# 6.3 Asymmetric target errors
fig, ax = plt.subplots(figsize=(10, 5))
order_idx = np.argsort(y)
ax.errorbar(np.arange(N), y[order_idx], yerr=[yerr_low_arr[order_idx], yerr_up_arr[order_idx]],
            fmt="o", ms=5, capsize=2, alpha=0.80)
ax.set_xlabel("Objects sorted by target metallicity")
ax.set_ylabel(r"$12+\log(\mathrm{O/H})$")
ax.set_title("Object-specific metallicity uncertainties")
display_and_save_fig(fig, "03_target_asymmetric_errors.png", "Target asymmetric errors")

# 6.4 Spearman correlation heatmap
corr_cols = feature_cols + ["Y_TARGET", "YERR_SYM"]
corr_df = model_df[corr_cols].copy()
spearman = corr_df.corr(method="spearman", min_periods=5)
save_table(spearman.reset_index().rename(columns={"index": "variable"}), "spearman_correlation_matrix")
display_df(spearman.reset_index().rename(columns={"index": "variable"}), title="Spearman correlation matrix")
fig, ax = plt.subplots(figsize=(9, 7))
if HAVE_SEABORN:
    sns.heatmap(spearman, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax,
                cbar_kws={"label": "Spearman correlation"})
else:
    im = ax.imshow(spearman.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(spearman.columns))); ax.set_xticklabels(spearman.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(spearman.index))); ax.set_yticklabels(spearman.index)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
ax.set_title("Spearman correlation heatmap")
display_and_save_fig(fig, "04_spearman_correlation_heatmap.png", "Spearman correlation heatmap")

# 6.5 Pairplot, wrapped so it never stops the analysis.
if HAVE_SEABORN:
    try:
        pair_cols = [c for c in ["MSTAR_LOG", "LOG_SFR", "LOG_OIII_EW", "LOG_L_FUV", "Y_TARGET"] if c in model_df.columns]
        pp = sns.pairplot(model_df[pair_cols], diag_kind="hist", corner=True,
                          plot_kws={"alpha": 0.75, "s": 45})
        pp.fig.suptitle("Pairplot of core variables", y=1.02)
        pp_path = FIG_DIR / "05_pairplot_core_variables.png"
        pp.fig.savefig(pp_path, dpi=220, bbox_inches="tight")
        log(f"Saved pairplot -> {pp_path}")
        if HAVE_IPYTHON:
            display(pp.fig)
        plt.close(pp.fig)
    except Exception as exc:
        log(f"Pairplot skipped safely: {repr(exc)}")

# 6.6 SFR-mass relation
if "MSTAR_LOG" in model_df.columns and "LOG_SFR" in model_df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(model_df["MSTAR_LOG"], model_df["LOG_SFR"], c=y, cmap="plasma", s=85, edgecolor="k", alpha=0.85)
    ax.set_xlabel(r"$\log_{10}(M_\ast/M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(1+\mathrm{SFR})$")
    ax.set_title("SFR--stellar mass relation; colour = adopted metallicity target")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$12+\log(\mathrm{O/H})$")
    display_and_save_fig(fig, "06_sfr_mass_relation.png", "SFR-mass relation")

# 6.7 MZR diagnostic
if "MSTAR_LOG" in model_df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    colour_var = model_df["LOG_SFR"] if "LOG_SFR" in model_df.columns else np.arange(N)
    sc = ax.scatter(model_df["MSTAR_LOG"], y, c=colour_var, cmap="viridis", s=75, edgecolor="k", alpha=0.80)
    ax.errorbar(model_df["MSTAR_LOG"], y, yerr=[yerr_low_arr, yerr_up_arr], fmt="none", ecolor="gray", alpha=0.50, capsize=2)
    ax.set_xlabel(r"$\log_{10}(M_\ast/M_\odot)$")
    ax.set_ylabel(r"Adopted $12+\log(\mathrm{O/H})$")
    ax.set_title("Mass--metallicity diagnostic with object-specific errors")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("LOG_SFR" if "LOG_SFR" in model_df.columns else "index")
    display_and_save_fig(fig, "07_mzr_with_errors.png", "MZR diagnostic")

# 6.8 FMR alpha diagnostic. It is diagnostic only, not a proof of physical FMR validity.
fmr_table = None
if "MSTAR_LOG" in model_df.columns and "LOG_SFR" in model_df.columns:
    alphas = np.round(np.linspace(0, 1.2, 61), 3)
    rows = []
    weights = 1.0 / np.maximum(yerr_sym_arr, 0.03) ** 2
    for alpha in alphas:
        mu = model_df["MSTAR_LOG"].to_numpy() - alpha * model_df["LOG_SFR"].to_numpy()
        valid = np.isfinite(mu) & np.isfinite(y) & np.isfinite(weights)
        if valid.sum() < 5:
            continue
        X_fmr = np.column_stack([np.ones(valid.sum()), mu[valid]])
        try:
            # Weighted least squares via closed form; robust to statsmodels absence.
            W = np.diag(weights[valid] / np.nanmedian(weights[valid]))
            beta_hat = np.linalg.pinv(X_fmr.T @ W @ X_fmr) @ (X_fmr.T @ W @ y[valid])
            pred = X_fmr @ beta_hat
            wrms = safe_weighted_rmse(y[valid], pred, weights[valid])
            rows.append({"alpha": alpha, "weighted_residual_rms": wrms, "slope": beta_hat[1]})
        except Exception:
            pass
    fmr_table = pd.DataFrame(rows)
    if not fmr_table.empty:
        save_table(fmr_table, "fmr_alpha_diagnostic")
        display_df(fmr_table, title="FMR alpha diagnostic")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fmr_table["alpha"], fmr_table["weighted_residual_rms"], marker="o", lw=2)
        best_row = fmr_table.loc[fmr_table["weighted_residual_rms"].idxmin()]
        ax.axvline(best_row["alpha"], color="black", linestyle="--", label=f"min at alpha={best_row['alpha']:.2f}")
        ax.set_xlabel(r"FMR-style parameter $\alpha$ in $\mu_\alpha=\log M_\ast-\alpha\log(1+\mathrm{SFR})$")
        ax.set_ylabel("Weighted residual RMS")
        ax.set_title("FMR-style diagnostic only; not a physical validation test")
        ax.legend()
        display_and_save_fig(fig, "08_fmr_alpha_diagnostic.png", "FMR alpha diagnostic")

# -------------------------
# 7. Repeated-CV baselines
# -------------------------
base_preprocess = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]

models = {
    "Mean only": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="mean"))]),
    "Linear": Pipeline(base_preprocess + [("model", LinearRegression())]),
    "RidgeCV": Pipeline(base_preprocess + [("model", RidgeCV(alphas=np.logspace(-4, 4, 41)))]),
    "Polynomial Ridge degree 2": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-4, 4, 41)))
    ]),
    "Bayesian Ridge": Pipeline(base_preprocess + [("model", BayesianRidge())]),
    "Huber robust": Pipeline(base_preprocess + [("model", HuberRegressor(max_iter=1000))]),
    "Random Forest": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                        random_state=RANDOM_SEED, n_jobs=-1))
    ]),
    "Extra Trees": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3,
                                      random_state=RANDOM_SEED, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(n_estimators=150, learning_rate=0.035,
                                            max_depth=2, random_state=RANDOM_SEED))
    ])
}

# Gaussian process baseline with scalar alpha only. Per-object alpha cannot be safely used
# inside a generic Pipeline across repeated CV because train-fold lengths change.
if RUN_GAUSSIAN_PROCESS_BASELINE and N <= 300:
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(len(feature_cols)), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-5, 1.0))
    models["Gaussian Process"] = Pipeline(base_preprocess + [
        ("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           alpha=1e-4, random_state=RANDOM_SEED,
                                           n_restarts_optimizer=2))
    ])

if RUN_OPTIONAL_MLP_SENSITIVITY:
    models["MLP sensitivity only"] = Pipeline(base_preprocess + [
        ("model", MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", alpha=0.05,
                               max_iter=2000, random_state=RANDOM_SEED,
                               early_stopping=True, validation_fraction=0.20))
    ])


def final_estimator_accepts_sample_weight(estimator):
    try:
        final = estimator
        if hasattr(estimator, "named_steps"):
            final = list(estimator.named_steps.values())[-1]
        sig = inspect.signature(final.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False


def fit_estimator_safely(estimator, X_train, y_train, weights=None):
    if FIT_BASELINES_WITH_SAMPLE_WEIGHT and weights is not None and final_estimator_accepts_sample_weight(estimator):
        try:
            if hasattr(estimator, "named_steps"):
                final_name = list(estimator.named_steps.keys())[-1]
                estimator.fit(X_train, y_train, **{f"{final_name}__sample_weight": weights})
            else:
                estimator.fit(X_train, y_train, sample_weight=weights)
            return estimator
        except Exception as exc:
            log(f"Weighted fit failed and unweighted fit will be used: {repr(exc)[:160]}")
    estimator.fit(X_train, y_train)
    return estimator

n_splits_effective = min(N_SPLITS_CV, max(2, N // 6))
rkf = RepeatedKFold(n_splits=n_splits_effective, n_repeats=N_REPEATS_CV, random_state=RANDOM_SEED)

cv_rows = []
oof_rows = []
fold_id = 0
log("\nStarting repeated K-fold baseline benchmarking...")

for train_idx, test_idx in rkf.split(X):
    fold_id += 1
    if fold_id == 1 or fold_id % 5 == 0:
        log(f"  CV fold-run {fold_id} / {n_splits_effective * N_REPEATS_CV}")
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    weights_train = 1.0 / np.maximum(yerr_sym_arr[train_idx], 0.03) ** 2
    weights_test = 1.0 / np.maximum(yerr_sym_arr[test_idx], 0.03) ** 2

    for model_name, model in models.items():
        estimator = clone(model)
        try:
            estimator = fit_estimator_safely(estimator, X_train, y_train, weights_train)
            pred = np.asarray(estimator.predict(X_test), dtype=float)
            if pred.shape[0] != len(y_test) or not np.all(np.isfinite(pred)):
                raise ValueError("non-finite or wrong-length predictions")
            rmse = safe_rmse(y_test, pred)
            mae = float(mean_absolute_error(y_test, pred))
            wrmse = safe_weighted_rmse(y_test, pred, weights_test)
            r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan
            cv_rows.append({
                "fold": fold_id, "model": model_name, "n_train": len(train_idx), "n_test": len(test_idx),
                "RMSE": rmse, "MAE": mae, "weighted_RMSE": wrmse, "R2": r2, "error": ""
            })
            for idx, obs, pr, err in zip(test_idx, y_test, pred, yerr_sym_arr[test_idx]):
                oof_rows.append({
                    "fold": fold_id, "row_index": int(idx), "model": model_name,
                    "observed": float(obs), "predicted": float(pr), "residual": float(obs - pr), "yerr": float(err)
                })
        except Exception as exc:
            cv_rows.append({
                "fold": fold_id, "model": model_name, "n_train": len(train_idx), "n_test": len(test_idx),
                "RMSE": np.nan, "MAE": np.nan, "weighted_RMSE": np.nan, "R2": np.nan,
                "error": repr(exc)[:500]
            })

cv_df = pd.DataFrame(cv_rows)
oof_df = pd.DataFrame(oof_rows)

valid_cv_df = cv_df[np.isfinite(cv_df["RMSE"])].copy()
if valid_cv_df.empty:
    save_table(cv_df, "repeated_cv_all_folds_baselines_FAILED")
    raise RuntimeError("All baseline models failed. Check the error column in repeated_cv_all_folds_baselines_FAILED.csv")

cv_summary = valid_cv_df.groupby("model", dropna=False).agg(
    folds=("RMSE", "count"),
    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
    MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
    weighted_RMSE_mean=("weighted_RMSE", "mean"), weighted_RMSE_sd=("weighted_RMSE", "std"),
    R2_mean=("R2", "mean"), R2_sd=("R2", "std")
).reset_index().sort_values("RMSE_mean")

save_table(cv_df, "repeated_cv_all_folds_baselines")
save_table(cv_summary, "repeated_cv_model_summary_baselines")
save_table(oof_df.head(500), "out_of_fold_predictions_baselines_head")
display_df(cv_summary, title="Repeated-CV model summary")

best_baseline_name = str(cv_summary.iloc[0]["model"])
log(f"Best non-Bayesian/sensitivity baseline by mean repeated-CV RMSE: {best_baseline_name}")

# CV benchmark bar plot
order = cv_summary.sort_values("RMSE_mean")["model"].tolist()
fig, ax = plt.subplots(figsize=(11, max(5, 0.50 * len(order))))
y_pos = np.arange(len(order))
summary_idx = cv_summary.set_index("model").loc[order]
ax.barh(y_pos, summary_idx["RMSE_mean"].to_numpy(), xerr=summary_idx["RMSE_sd"].fillna(0).to_numpy(),
        capsize=3, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_title("Repeated K-fold benchmark: RMSE mean ± SD")
ax.set_xlabel("RMSE in target units")
ax.set_ylabel("")
display_and_save_fig(fig, "09_cv_model_benchmark_rmse_baselines.png", "Baseline CV benchmark")

# CV RMSE distribution with pure matplotlib, robust to seaborn bugs/NaNs.
fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(order))))
box_data = [valid_cv_df.loc[valid_cv_df["model"] == m, "RMSE"].dropna().to_numpy() for m in order]
positions = np.arange(1, len(order) + 1)
ax.boxplot(box_data, vert=False, positions=positions, labels=order, patch_artist=True, showmeans=True)
ax.invert_yaxis()
ax.set_title("Distribution of RMSE across repeated folds")
ax.set_xlabel("RMSE")
ax.set_ylabel("")
display_and_save_fig(fig, "10_cv_rmse_distribution_baselines.png", "Baseline CV RMSE distribution")

# Best baseline OOF observed vs predicted.
best_oof = oof_df[oof_df["model"] == best_baseline_name].copy()
if not best_oof.empty:
    avg_best_oof = best_oof.groupby("row_index").agg(
        observed=("observed", "mean"),
        predicted=("predicted", "mean"),
        pred_sd=("predicted", "std"),
        yerr=("yerr", "mean")
    ).reset_index()
    save_table(avg_best_oof, "best_baseline_oof_predictions")
    display_df(avg_best_oof, title="Best baseline OOF predictions")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.errorbar(avg_best_oof["observed"], avg_best_oof["predicted"],
                xerr=avg_best_oof["yerr"], yerr=avg_best_oof["pred_sd"].fillna(0),
                fmt="o", alpha=0.85, capsize=2)
    lo = min(avg_best_oof["observed"].min(), avg_best_oof["predicted"].min())
    hi = max(avg_best_oof["observed"].max(), avg_best_oof["predicted"].max())
    pad = 0.05 * (hi - lo + 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.5)
    ax.set_xlabel("Observed target")
    ax.set_ylabel("Out-of-fold predicted target")
    ax.set_title(f"OOF observed vs predicted: {best_baseline_name}")
    display_and_save_fig(fig, "11_oof_observed_vs_predicted_best_baseline.png", "OOF observed vs predicted")

# Fit best baseline on all data for diagnostics only.
best_est = clone(models[best_baseline_name])
weights_all = 1.0 / np.maximum(yerr_sym_arr, 0.03) ** 2
best_est = fit_estimator_safely(best_est, X, y, weights_all)
joblib.dump(best_est, MODEL_DIR / "best_baseline_model.joblib")

pred_all = np.asarray(best_est.predict(X), dtype=float)
resid_all = y - pred_all
diag_df = pd.DataFrame({"observed": y, "predicted_all_data": pred_all, "residual": resid_all, "yerr": yerr_sym_arr})
save_table(diag_df, "best_baseline_all_data_diagnostics")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(pred_all, resid_all, c=y, cmap="coolwarm", s=80, edgecolor="k", alpha=0.85)
ax.axhline(0, color="black", linestyle="--")
ax.set_xlabel("All-data fitted value")
ax.set_ylabel("Residual")
ax.set_title(f"Residuals vs fitted values: {best_baseline_name}\n(diagnostic only; use CV for performance)")
fig.colorbar(sc, ax=ax, label="Observed target")
display_and_save_fig(fig, "12_residuals_vs_fitted_best_baseline.png", "Residuals vs fitted")

fig, ax = plt.subplots(figsize=(8, 5))
stats.probplot(resid_all[np.isfinite(resid_all)], dist="norm", plot=ax)
ax.set_title("Normal Q-Q plot of all-data residuals")
display_and_save_fig(fig, "13_residuals_qq_best_baseline.png", "Residual Q-Q")

# Residuals vs key predictors
for var, label in [("MSTAR_LOG", r"$\log_{10}(M_\ast/M_\odot)$"), ("LOG_SFR", r"$\log_{10}(1+\mathrm{SFR})$"), ("LOG_OIII_EW", r"$\log_{10}(1+\mathrm{EW}_{[OIII]})$")]:
    if var in model_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(model_df[var], resid_all, c=y, cmap="viridis", s=75, edgecolor="k", alpha=0.85)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals vs {var}: {best_baseline_name}")
        display_and_save_fig(fig, f"14_residuals_vs_{var.lower()}.png", f"Residuals vs {var}")

# Permutation importance, if estimator supports scoring.
try:
    perm = permutation_importance(best_est, X, y, n_repeats=80, random_state=RANDOM_SEED,
                                  scoring="neg_root_mean_squared_error")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_sd": perm.importances_std
    }).sort_values("importance_mean", ascending=False)
    save_table(imp_df, "permutation_importance_best_baseline")
    display_df(imp_df, title="Permutation importance for best all-data baseline")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["feature"], imp_df["importance_mean"], xerr=imp_df["importance_sd"], alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Decrease in negative RMSE score when permuted")
    ax.set_title(f"Permutation importance: {best_baseline_name}\n(diagnostic only)")
    display_and_save_fig(fig, "15_permutation_importance_best_baseline.png", "Permutation importance")
except Exception as exc:
    log(f"Permutation importance skipped safely: {repr(exc)}")

# -------------------------
# 8. Heteroscedastic Bayesian calibration model
# -------------------------
bayes_summary = None
bayes_ppc_df = None

if RUN_BAYESIAN_MODEL:
    try:
        import pymc as pm
        import arviz as az
        HAVE_PYMC = True
    except Exception as exc:
        HAVE_PYMC = False
        log(f"PyMC/ArviZ not available; Bayesian model skipped safely: {repr(exc)}")

    if HAVE_PYMC:
        log("\nFitting heteroscedastic Bayesian calibration model on full data for posterior diagnostics...")
        imputer_bayes = SimpleImputer(strategy="median")
        scaler_bayes = StandardScaler()
        X_imp = imputer_bayes.fit_transform(X)
        X_scaled = scaler_bayes.fit_transform(X_imp)
        p = X_scaled.shape[1]
        coords = {"feature": feature_cols, "obs_id": np.arange(N)}
        try:
            with pm.Model(coords=coords) as bayes_model:
                X_data = pm.Data("X", X_scaled, dims=("obs_id", "feature"))
                yerr_data = pm.Data("yerr", yerr_sym_arr, dims="obs_id")
                beta0 = pm.Normal("beta0", mu=float(np.mean(y)), sigma=1.0)
                beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")
                sigma_int = pm.HalfNormal("sigma_int", sigma=0.20)
                mu = pm.Deterministic("mu", beta0 + pm.math.dot(X_data, beta), dims="obs_id")
                sigma_tot = pm.Deterministic("sigma_tot", pm.math.sqrt(yerr_data**2 + sigma_int**2), dims="obs_id")
                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_tot, observed=y, dims="obs_id")
                idata = pm.sample(draws=1000, tune=1000, chains=2, cores=1,
                                  target_accept=0.92, random_seed=RANDOM_SEED,
                                  return_inferencedata=True, progressbar=True)
                ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs", "mu"],
                                                     random_seed=RANDOM_SEED, progressbar=False)
            az.to_netcdf(idata, MODEL_DIR / "heteroscedastic_bayesian_model_idata.nc")
            joblib.dump({"imputer": imputer_bayes, "scaler": scaler_bayes, "features": feature_cols},
                        MODEL_DIR / "bayesian_preprocess.joblib")

            bayes_summary = az.summary(idata, var_names=["beta0", "beta", "sigma_int"], hdi_prob=0.94).reset_index()
            save_table(bayes_summary, "bayesian_posterior_summary")
            display_df(bayes_summary, title="Bayesian posterior summary")

            # Trace plot
            try:
                axes = az.plot_trace(idata, var_names=["beta0", "beta", "sigma_int"], compact=True)
                fig = plt.gcf()
                fig.suptitle("Trace plots for heteroscedastic Bayesian calibration model", y=1.02)
                display_and_save_fig(fig, "16_bayesian_trace_plots.png", "Bayesian trace plots")
            except Exception as exc:
                log(f"Trace plot skipped safely: {repr(exc)}")

            # Posterior predictive arrays, robust to version shapes.
            yrep = ppc.posterior_predictive["y_obs"].values.reshape(-1, N)
            mu_draws = ppc.posterior_predictive["mu"].values.reshape(-1, N) if "mu" in ppc.posterior_predictive else idata.posterior["mu"].values.reshape(-1, N)
            yrep_mean = yrep.mean(axis=0)
            yrep_lo = np.percentile(yrep, 3, axis=0)
            yrep_hi = np.percentile(yrep, 97, axis=0)
            mu_mean = mu_draws.mean(axis=0)
            mu_lo = np.percentile(mu_draws, 3, axis=0)
            mu_hi = np.percentile(mu_draws, 97, axis=0)
            bayes_ppc_df = pd.DataFrame({
                "observed": y,
                "mu_mean": mu_mean,
                "mu_3pct": mu_lo,
                "mu_97pct": mu_hi,
                "posterior_predictive_mean": yrep_mean,
                "posterior_predictive_3pct": yrep_lo,
                "posterior_predictive_97pct": yrep_hi,
                "residual_mu": y - mu_mean,
                "yerr": yerr_sym_arr,
                "covered_94pct_predictive": (y >= yrep_lo) & (y <= yrep_hi)
            })
            save_table(bayes_ppc_df, "bayesian_posterior_predictive_checks")
            display_df(bayes_ppc_df, title="Bayesian posterior predictive checks")

            coverage = float(bayes_ppc_df["covered_94pct_predictive"].mean())
            log(f"Empirical 94% posterior predictive interval coverage: {coverage:.3f}")

            # Observed vs Bayesian posterior mean with predictive intervals.
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.errorbar(y, mu_mean,
                        xerr=yerr_sym_arr,
                        yerr=[mu_mean - mu_lo, mu_hi - mu_mean],
                        fmt="o", alpha=0.85, capsize=2)
            lo = min(np.min(y), np.min(mu_lo))
            hi = max(np.max(y), np.max(mu_hi))
            pad = 0.05 * (hi - lo + 1e-6)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.5)
            ax.set_xlabel("Observed target")
            ax.set_ylabel("Bayesian posterior mean")
            ax.set_title("Bayesian heteroscedastic calibration: observed vs posterior mean")
            display_and_save_fig(fig, "17_bayesian_observed_vs_posterior_mean.png", "Bayesian observed vs posterior mean")

            # Posterior predictive distribution overlay.
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(y, bins=min(14, max(6, int(np.sqrt(N)))), alpha=0.65, density=True, label="observed")
            ax.hist(yrep.reshape(-1), bins=25, alpha=0.40, density=True, label="posterior predictive")
            ax.set_xlabel(r"Target $12+\log(\mathrm{O/H})$")
            ax.set_ylabel("Density")
            ax.set_title("Posterior predictive check: distribution")
            ax.legend()
            display_and_save_fig(fig, "18_bayesian_ppc_distribution.png", "Bayesian PPC distribution")

            # Residuals vs posterior mean.
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(mu_mean, y - mu_mean, c=yerr_sym_arr, cmap="magma", s=80, edgecolor="k", alpha=0.85)
            ax.axhline(0, color="black", linestyle="--")
            ax.set_xlabel("Posterior mean")
            ax.set_ylabel("Observed - posterior mean")
            ax.set_title("Bayesian residual diagnostic")
            cb = fig.colorbar(ax.collections[0], ax=ax)
            cb.set_label("Observed metallicity uncertainty")
            display_and_save_fig(fig, "19_bayesian_residuals_vs_posterior_mean.png", "Bayesian residuals")

        except Exception as exc:
            log(f"Bayesian model failed but the rest of the pipeline is complete. Error: {repr(exc)}")
            (REPORT_DIR / "bayesian_model_error.txt").write_text(repr(exc))

# Optional Bayesian CV placeholder: implemented conservatively as not run by default.
if RUN_BAYESIAN_CV:
    log("RUN_BAYESIAN_CV=True requested, but full repeated Bayesian CV is intentionally not run in this compact script because it is slow. Use the saved baseline CV as the main validation and the full Bayesian model for uncertainty diagnostics.")

# -------------------------
# 9. PCA/UMAP exploratory visualization only
# -------------------------
try:
    from sklearn.decomposition import PCA
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "Y_TARGET": y})
    save_table(pca_df, "pca_embedding")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="plasma", s=80, edgecolor="k", alpha=0.85)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.set_title("PCA embedding of predictor space\n(exploratory only)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Adopted $12+\log(\mathrm{O/H})$")
    display_and_save_fig(fig, "20_pca_embedding_exploratory.png", "PCA exploratory embedding")
except Exception as exc:
    log(f"PCA skipped safely: {repr(exc)}")

try:
    import umap
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    reducer = umap.UMAP(n_neighbors=min(12, max(4, N // 5)), min_dist=0.15, random_state=RANDOM_SEED)
    U = reducer.fit_transform(X_scaled)
    umap_df = pd.DataFrame({"UMAP1": U[:, 0], "UMAP2": U[:, 1], "Y_TARGET": y})
    save_table(umap_df, "umap_embedding")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(U[:, 0], U[:, 1], c=y, cmap="viridis", s=80, edgecolor="k", alpha=0.85)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP embedding of predictor space\n(exploratory only; not evidence for subpopulations)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Adopted $12+\log(\mathrm{O/H})$")
    display_and_save_fig(fig, "21_umap_embedding_exploratory.png", "UMAP exploratory embedding")
except Exception as exc:
    log(f"UMAP skipped safely: {repr(exc)}")

# -------------------------
# 10. Auto-generated README/report and zip
# -------------------------
summary_text = f"""
# Green Pea revised analysis outputs v3

## Dataset
- Loaded file: `{data_path}`
- Raw shape: {raw_df.shape}
- Modelling sample size: N = {N}
- Target column: `{target_col}`
- Target interpretation: `{target_kind}`
- Predictors used: {', '.join(feature_cols)}

## Scientific interpretation warning
If the target interpretation is `strong_line_calibration`, the results quantify empirical
reproduction of the adopted metallicity calibration scale. They must not be described as
recovery of direct physical oxygen abundances.

## Main validation
The main predictive validation is repeated K-fold cross-validation of baseline/flexible models.
The best model by mean RMSE in this run was:

- `{best_baseline_name}`

See:
- `tables/repeated_cv_all_folds_baselines.csv`
- `tables/repeated_cv_model_summary_baselines.csv`
- `figures/09_cv_model_benchmark_rmse_baselines.png`
- `figures/10_cv_rmse_distribution_baselines.png`

## Bayesian model
The heteroscedastic Bayesian calibration model is fitted on the full data for posterior diagnostics,
using object-specific metallicity errors through
sigma_total_i = sqrt(yerr_i^2 + sigma_int^2).
This is not used as a single train-test performance claim.

## Deleted/avoided old-paper mistakes
- No [Fe/H] figure is produced.
- No negative metallicity colour scale is produced.
- Dimensionality reduction is marked exploratory only.
- Low RMSE is interpreted as calibration-scale reproducibility, not true abundance recovery.

## Code fixes in v3
- Removed `mean_squared_error(..., squared=False)` because some Colab/sklearn versions reject it.
- Gaussian Process baseline now uses scalar alpha inside CV, avoiding train-fold length mismatch.
- Boxplots use pure matplotlib and filter finite RMSE values, avoiding seaborn `boxprops` crashes.
- Baseline fitting does not pass sample weights by default, avoiding estimator-specific keyword errors.
""".strip()

readme_path = OUT / "README_outputs_v3.md"
readme_path.write_text(summary_text)
log(f"Saved output README -> {readme_path}")

# Save run log.
(REPORT_DIR / "run_log.txt").write_text("\n".join(LOG_LINES))

# Zip all outputs.
zip_path = Path(f"{OUTPUT_DIR}.zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUT.rglob("*"):
        if path.is_file():
            zf.write(path, arcname=str(path.relative_to(OUT.parent)))
log(f"\nCreated ZIP archive: {zip_path.resolve()}")

# In Colab, offer automatic download.
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    pass

log("\nAnalysis complete. Use the ZIP, tables, and figures from green_pea_revised_outputs_v3.")
