# ============================================================
# AIDO-D Plan B Part 2 — HNSC
# Fixed-path version for:
#   D:/AIDO-Data/UCSC_XENA/Head and Neck Cancer (HNSC)
#
# Input:
#   GE.tsv
#   Phenotype.tsv
#   h.all.v2026.1.Hs.symbols.gmt
#
# Optional full Plan B GMT files, if available:
#   D:/AIDO-Data/GSEA/c5.go.bp.v2026.1.Hs.symbols.gmt
#   D:/AIDO-Data/GSEA/c2.cp.reactome.v2026.1.Hs.symbols.gmt
#
# Output:
#   D:/AIDO-Temp/AIDO-D-PlanB-HNSC
# ============================================================

import re
import math
import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False


# ============================================================
# 0. Fixed configuration
# ============================================================

CANCER = "BRCA"

DATA_ROOT = Path(r"D:\AIDO-Data\UCSC_XENA\Breast Cancer (BRCA)")
GSEA_ROOT = Path(r"D:/AIDO-Data/GSEA")
OUT_ROOT = Path(r"D:/AIDO-Temp/AIDO-D-PlanB-BRCA")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPR_FILE = DATA_ROOT / "GE.tsv"
SURV_FILE = DATA_ROOT / "Phenotype.tsv"

GMT_FILES = {
    "Hallmark": DATA_ROOT / "h.all.v2026.1.Hs.symbols.gmt",
    "GO_BP": GSEA_ROOT / "c5.go.bp.v2026.1.Hs.symbols.gmt",
    "Reactome": GSEA_ROOT / "c2.cp.reactome.v2026.1.Hs.symbols.gmt",
}

MIN_MAPPED_GENES = 5
CONTROL_MIN_GENES = 10
CONTROL_MAX_GENES = 300

FWHM_METHOD = "kde"
DELTA = 1e-6

LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
K_VALUES = [20, 50, 100, 200, 500]

DPI = 300


# ============================================================
# 1. Basic utilities
# ============================================================

def log(msg):
    print(f"[AIDO-D Plan B | {CANCER}] {msg}")


def read_table_auto(path):
    """
    Robust reader for TCGA/Xena tables.
    Handles UTF-8, UTF-8-SIG, UTF-16, UTF-16LE, Latin-1 encodings.
    GBM Phenotype.tsv may be UTF-16LE, causing:
        UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff...
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    compression = "gzip" if str(path).endswith(".gz") else None
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "latin1"]
    seps = ["\t", ","]

    last_error = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    compression=compression,
                    encoding=enc,
                    low_memory=False
                )
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Cannot read file: {path}. Last error: {last_error}")


def normalize_tcga_barcode(x, n=12):
    x = str(x).strip()
    return x[:n] if len(x) >= n else x


def is_primary_tumor_barcode(x):
    x = str(x)
    parts = x.split("-")
    if len(parts) >= 4:
        return parts[3][:2] == "01"
    return True


def standardize_gene_symbol(x):
    x = str(x).strip()
    if "|" in x:
        x = x.split("|")[0]
    x = re.sub(r"\.\d+$", "", x)
    return x.upper()


# ============================================================
# 2. Load expression
# ============================================================

def load_expression_matrix(path):
    log(f"Loading expression file: {path}")
    df = read_table_auto(path)

    gene_col = df.columns[0]
    for c in df.columns[:5]:
        cl = str(c).lower()
        if cl in ["gene", "genes", "gene_symbol", "symbol", "hugo_symbol"]:
            gene_col = c
            break

    df = df.rename(columns={gene_col: "Gene"})
    df["Gene"] = df["Gene"].astype(str).map(standardize_gene_symbol)

    sample_cols = [c for c in df.columns if c != "Gene"]

    expr = df[["Gene"] + sample_cols].copy()
    for c in sample_cols:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")

    expr = expr.dropna(subset=["Gene"])
    expr = expr[expr["Gene"] != ""]
    expr = expr.groupby("Gene", as_index=True).mean(numeric_only=True)

    primary_cols = [c for c in expr.columns if is_primary_tumor_barcode(str(c))]
    if len(primary_cols) > 0:
        expr = expr[primary_cols]

    patient_ids = [normalize_tcga_barcode(c, 12) for c in expr.columns]
    expr.columns = patient_ids

    expr = expr.T.groupby(level=0).mean().T

    expr = expr.loc[expr.notna().mean(axis=1) >= 0.5]
    expr = expr.apply(lambda row: row.fillna(row.median()), axis=1)

    log(f"Expression loaded: {expr.shape[0]} genes x {expr.shape[1]} patients")
    return expr


def zscore_genes(expr):
    mean = expr.mean(axis=1)
    std = expr.std(axis=1).replace(0, np.nan)
    z = expr.sub(mean, axis=0).div(std, axis=0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z


# ============================================================
# 3. Load GMT gene sets
# ============================================================

def load_gmt(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GMT file not found: {path}")

    gene_sets = {}
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = sorted(set(standardize_gene_symbol(g) for g in parts[2:] if g.strip()))
            gene_sets[name] = genes

    return gene_sets


# ============================================================
# 4. Load survival / phenotype
# ============================================================

def load_survival(path):
    log(f"Loading survival/phenotype file: {path}")
    df = read_table_auto(path)
    df.columns = [str(c).strip() for c in df.columns]

    id_candidates = [
        "sample", "Sample", "patient", "Patient", "PATIENT",
        "submitter_id", "bcr_patient_barcode", "barcode", "ID"
    ]

    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    df["Patient"] = df[id_col].astype(str).map(lambda x: normalize_tcga_barcode(x, 12))

    time_candidates = [
        "OS.time", "OS_Time", "OS_TIME", "OS_Time_nature2012",
        "OS.time.days", "days_to_death", "days_to_last_followup",
        "PFI.time", "DSS.time"
    ]

    event_candidates = [
        "OS", "OS_event", "OS_EVENT", "OS_event_nature2012",
        "vital_status", "PFI", "DSS"
    ]

    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        death_cols = [c for c in df.columns if "death" in c.lower() and "day" in c.lower()]
        follow_cols = [c for c in df.columns if ("follow" in c.lower() or "last" in c.lower()) and "day" in c.lower()]
        if death_cols or follow_cols:
            death = pd.to_numeric(df[death_cols[0]], errors="coerce") if death_cols else pd.Series(np.nan, index=df.index)
            follow = pd.to_numeric(df[follow_cols[0]], errors="coerce") if follow_cols else pd.Series(np.nan, index=df.index)
            df["_time_auto"] = death.fillna(follow)
            time_col = "_time_auto"

    if time_col is None:
        numeric_cols = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0 and s.max(skipna=True) > 30:
                numeric_cols.append(c)
        if len(numeric_cols) > 0:
            time_col = numeric_cols[0]

    event_col = None
    for c in event_candidates:
        if c in df.columns:
            event_col = c
            break

    if event_col is None:
        vital_cols = [c for c in df.columns if "vital" in c.lower() or "status" in c.lower()]
        if len(vital_cols) > 0:
            event_col = vital_cols[0]

    if time_col is None or event_col is None:
        print("\nAvailable columns in Phenotype.tsv:")
        print(list(df.columns))
        raise ValueError("Cannot identify survival time/event columns. Please inspect columns above.")

    time = pd.to_numeric(df[time_col], errors="coerce")

    raw_event = df[event_col]
    if raw_event.dtype == object:
        event = raw_event.astype(str).str.lower().map({
            "dead": 1, "deceased": 1, "1": 1, "true": 1, "yes": 1,
            "alive": 0, "living": 0, "0": 0, "false": 0, "no": 0
        })
        event = event.fillna(pd.to_numeric(raw_event, errors="coerce"))
    else:
        event = pd.to_numeric(raw_event, errors="coerce")

    surv = pd.DataFrame({
        "Patient": df["Patient"],
        "time": time,
        "event": event
    })

    surv = surv.dropna(subset=["Patient", "time", "event"])
    surv = surv[surv["time"] > 0]
    surv["event"] = surv["event"].astype(int)
    surv = surv.drop_duplicates("Patient")

    log(f"Survival loaded: {surv.shape[0]} patients")
    log(f"Using time column: {time_col}")
    log(f"Using event column: {event_col}")

    return surv


# ============================================================
# 5. FWHM
# ============================================================

def empirical_fwhm(values, method="kde"):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3:
        return np.nan

    if np.nanstd(x) < 1e-12:
        return 0.0

    if method == "kde" and SCIPY_AVAILABLE and len(x) >= 5:
        try:
            kde = gaussian_kde(x)
            sd = np.std(x)
            grid = np.linspace(np.min(x) - 0.25 * sd, np.max(x) + 0.25 * sd, 256)
            y = kde(grid)
            half_max = np.max(y) / 2.0
            above = grid[y >= half_max]
            if len(above) >= 2:
                return float(above[-1] - above[0])
        except Exception:
            pass

    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    return float(2.355 * iqr / 1.349)


# ============================================================
# 6. Compute gene-set μ and W matrices
# ============================================================

def compute_mu_w_for_database(zexpr, gene_sets, db_name):
    genes_available = set(zexpr.index)
    patients = list(zexpr.columns)

    mu_records = []
    w_records = []
    summary_records = []

    total = len(gene_sets)
    kept = 0

    for i, (gs_name, genes) in enumerate(gene_sets.items(), start=1):
        mapped = sorted(set(genes).intersection(genes_available))
        n_mapped = len(mapped)

        if n_mapped < MIN_MAPPED_GENES:
            continue

        mat = zexpr.loc[mapped, patients]

        mu = mat.mean(axis=0)

        arr = mat.values
        w_values = []
        for j in range(arr.shape[1]):
            w_values.append(empirical_fwhm(arr[:, j], method=FWHM_METHOD))

        w = pd.Series(w_values, index=patients)

        mu_records.append(pd.Series(mu.values, index=patients, name=gs_name))
        w_records.append(pd.Series(w.values, index=patients, name=gs_name))

        summary_records.append({
            "database": db_name,
            "gene_set": gs_name,
            "n_genes_original": len(set(genes)),
            "n_genes_mapped": n_mapped,
            "mu_mean_abs": float(np.nanmean(np.abs(mu.values))),
            "mu_mean": float(np.nanmean(mu.values)),
            "W_mean": float(np.nanmean(w.values)),
            "W_median": float(np.nanmedian(w.values)),
            "W_sd": float(np.nanstd(w.values)),
            "coherence_mean": float(np.nanmean(np.abs(mu.values) / (w.values + DELTA))),
        })

        kept += 1

        if i % 500 == 0:
            log(f"{db_name}: processed {i}/{total}, kept {kept}")

    mu_matrix = pd.DataFrame(mu_records)
    w_matrix = pd.DataFrame(w_records)
    summary = pd.DataFrame(summary_records)

    log(f"{db_name}: retained {mu_matrix.shape[0]} gene sets x {mu_matrix.shape[1]} patients")
    return mu_matrix, w_matrix, summary


# ============================================================
# 7. Survival analysis
# ============================================================

def survival_logrank(score, surv, label, out_png=None):
    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines is required. Install by: pip install lifelines")

    tmp = pd.DataFrame({
        "Patient": score.index,
        "score": score.values
    })

    tmp = tmp.merge(surv, on="Patient", how="inner")
    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=["score", "time", "event"])

    if tmp.shape[0] < 20 or tmp["event"].sum() < 5:
        return {
            "label": label,
            "n": int(tmp.shape[0]),
            "events": int(tmp["event"].sum()) if tmp.shape[0] > 0 else 0,
            "p": np.nan,
            "D_clinical": np.nan
        }

    median = tmp["score"].median()
    tmp["group"] = np.where(tmp["score"] >= median, "High", "Low")

    high = tmp[tmp["group"] == "High"]
    low = tmp[tmp["group"] == "Low"]

    if high.shape[0] < 5 or low.shape[0] < 5:
        return {
            "label": label,
            "n": int(tmp.shape[0]),
            "events": int(tmp["event"].sum()),
            "p": np.nan,
            "D_clinical": np.nan
        }

    res = logrank_test(
        high["time"], low["time"],
        event_observed_A=high["event"],
        event_observed_B=low["event"]
    )

    p = float(res.p_value)
    D = -math.log10(max(p, 1e-300))

    if out_png is not None:
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(5.4, 4.3))

        kmf.fit(low["time"], low["event"], label=f"Low {label} (n={low.shape[0]})")
        kmf.plot_survival_function(ci_show=False)

        kmf.fit(high["time"], high["event"], label=f"High {label} (n={high.shape[0]})")
        kmf.plot_survival_function(ci_show=False)

        plt.title(f"{CANCER}: {label}\nlog-rank p={p:.3e}, D={D:.2f}")
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.tight_layout()
        plt.savefig(out_png, dpi=DPI)
        plt.close()

    return {
        "label": label,
        "n": int(tmp.shape[0]),
        "events": int(tmp["event"].sum()),
        "median": float(median),
        "p": p,
        "D_clinical": D,
        "high_n": int(high.shape[0]),
        "low_n": int(low.shape[0])
    }


# ============================================================
# 8. Main workflow
# ============================================================

def main():
    if not LIFELINES_AVAILABLE:
        raise ImportError("Please install lifelines first: pip install lifelines")

    log("Starting BRCA Plan B workflow")
    log(f"DATA_ROOT = {DATA_ROOT}")
    log(f"EXPR_FILE = {EXPR_FILE}")
    log(f"SURV_FILE = {SURV_FILE}")

    if not EXPR_FILE.exists():
        raise FileNotFoundError(f"Expression file not found: {EXPR_FILE}")

    if not SURV_FILE.exists():
        raise FileNotFoundError(f"Phenotype/survival file not found: {SURV_FILE}")

    expr = load_expression_matrix(EXPR_FILE)
    zexpr = zscore_genes(expr)

    surv = load_survival(SURV_FILE)

    common = sorted(set(zexpr.columns).intersection(set(surv["Patient"])))
    if len(common) < 20:
        raise RuntimeError(f"Too few matched patients: {len(common)}")

    zexpr = zexpr[common]
    surv = surv[surv["Patient"].isin(common)].copy()

    log(f"Matched cohort: {len(common)} patients")

    all_mu = []
    all_w = []
    all_summary = []

    for db_name, gmt_path in GMT_FILES.items():
        if not gmt_path.exists():
            log(f"WARNING: missing GMT skipped: {gmt_path}")
            continue

        log(f"Loading {db_name}: {gmt_path}")
        gene_sets = load_gmt(gmt_path)
        log(f"{db_name}: {len(gene_sets)} gene sets loaded")

        mu, w, summary = compute_mu_w_for_database(zexpr, gene_sets, db_name)

        db_out = OUT_ROOT / db_name
        db_out.mkdir(parents=True, exist_ok=True)

        mu.to_csv(db_out / f"{CANCER}_{db_name}_mu_matrix.csv")
        w.to_csv(db_out / f"{CANCER}_{db_name}_FWHM_W_matrix.csv")
        summary.to_csv(db_out / f"{CANCER}_{db_name}_gene_set_summary.csv", index=False)

        mu.index = pd.MultiIndex.from_product([[db_name], mu.index], names=["database", "gene_set"])
        w.index = pd.MultiIndex.from_product([[db_name], w.index], names=["database", "gene_set"])

        all_mu.append(mu)
        all_w.append(w)
        all_summary.append(summary)

    if len(all_summary) == 0:
        raise RuntimeError("No GMT database was processed. Check GMT paths.")

    mu_all = pd.concat(all_mu)
    w_all = pd.concat(all_w)
    summary_all = pd.concat(all_summary, ignore_index=True)

    mu_all.to_csv(OUT_ROOT / f"{CANCER}_ALL_mu_matrix.csv")
    w_all.to_csv(OUT_ROOT / f"{CANCER}_ALL_FWHM_W_matrix.csv")
    summary_all.to_csv(OUT_ROOT / f"{CANCER}_ALL_gene_set_summary.csv", index=False)

    # ------------------------------------------------------------
    # Global μ-W state-cloud map
    # ------------------------------------------------------------

    log("Generating global μ-W state-cloud map")

    plt.figure(figsize=(6.5, 5.2))
    for db in summary_all["database"].unique():
        sub = summary_all[summary_all["database"] == db]
        plt.scatter(sub["mu_mean_abs"], sub["W_mean"], s=10, alpha=0.55, label=db)

    plt.xlabel("Mean |μB| across patients")
    plt.ylabel("Mean FWHM W_B across patients")
    plt.title(f"{CANCER}: Gene-set mean–FWHM state-cloud map")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"{CANCER}_global_mu_FWHM_state_cloud_map.png", dpi=DPI)
    plt.close()

    plt.figure(figsize=(6.5, 5.2))
    for db in summary_all["database"].unique():
        sub = summary_all[summary_all["database"] == db]
        plt.scatter(sub["n_genes_mapped"], sub["W_mean"], s=10, alpha=0.55, label=db)

    plt.xscale("log")
    plt.xlabel("Mapped gene-set size")
    plt.ylabel("Mean FWHM W_B")
    plt.title(f"{CANCER}: Gene-set size vs FWHM")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"{CANCER}_gene_set_size_vs_FWHM.png", dpi=DPI)
    plt.close()

    summary_all.sort_values("W_mean", ascending=True).head(100).to_csv(
        OUT_ROOT / f"{CANCER}_top100_narrow_low_W_gene_sets.csv", index=False
    )

    summary_all.sort_values("W_mean", ascending=False).head(100).to_csv(
        OUT_ROOT / f"{CANCER}_top100_broad_high_W_gene_sets.csv", index=False
    )

    summary_all.sort_values("coherence_mean", ascending=False).head(100).to_csv(
        OUT_ROOT / f"{CANCER}_top100_coherent_gene_sets.csv", index=False
    )

    # ------------------------------------------------------------
    # Patient-level D_W-PHY
    # ------------------------------------------------------------

    log("Computing patient-level D_W-PHY")

    M_all = mu_all.abs().mean(axis=0)
    U_all = np.log(w_all + DELTA).mean(axis=0)

    patient_df = pd.DataFrame({
        "Patient": M_all.index,
        "M_mean_abs_mu": M_all.values,
        "U_mean_log_W": U_all.values
    }).set_index("Patient")

    d_results = []

    for lam in LAMBDA_GRID:
        score = patient_df["M_mean_abs_mu"] - lam * patient_df["U_mean_log_W"]
        patient_df[f"D_W_PHY_lambda_{lam}"] = score

        res = survival_logrank(
            score,
            surv,
            label=f"D_W-PHY lambda={lam}",
            out_png=OUT_ROOT / f"{CANCER}_KM_D_W_PHY_lambda_{lam}.png"
        )

        res["lambda"] = lam
        d_results.append(res)

    d_results_df = pd.DataFrame(d_results)
    d_results_df.to_csv(OUT_ROOT / f"{CANCER}_D_W_PHY_lambda_scan_survival.csv", index=False)
    patient_df.reset_index().to_csv(OUT_ROOT / f"{CANCER}_patient_level_D_W_PHY_scores.csv", index=False)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(d_results_df["lambda"], d_results_df["D_clinical"], marker="o")
    plt.xlabel("lambda")
    plt.ylabel("Clinical discriminability: -log10(p)")
    plt.title(f"{CANCER}: D_W-PHY lambda scan")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"{CANCER}_D_W_PHY_lambda_scan.png", dpi=DPI)
    plt.close()

    # ------------------------------------------------------------
    # Controlled map: 10 <= mapped genes <= 300
    # ------------------------------------------------------------

    controlled = summary_all[
        (summary_all["n_genes_mapped"] >= CONTROL_MIN_GENES) &
        (summary_all["n_genes_mapped"] <= CONTROL_MAX_GENES)
    ].copy()

    controlled.to_csv(OUT_ROOT / f"{CANCER}_controlled_gene_sets_10_to_300.csv", index=False)

    plt.figure(figsize=(6.5, 5.2))
    for db in controlled["database"].unique():
        sub = controlled[controlled["database"] == db]
        plt.scatter(sub["mu_mean_abs"], sub["W_mean"], s=10, alpha=0.55, label=db)

    plt.xlabel("Mean |μB| across patients")
    plt.ylabel("Mean FWHM W_B across patients")
    plt.title(f"{CANCER}: Controlled μ-W map")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"{CANCER}_controlled_mu_FWHM_state_cloud_map.png", dpi=DPI)
    plt.close()

    # ------------------------------------------------------------
    # K-scale observation lens
    # ------------------------------------------------------------

    log("Computing K-scale observation lens")

    ranked = summary_all.sort_values("coherence_mean", ascending=False).copy()

    k_records = []

    for K in K_VALUES:
        selected = ranked.head(K)
        idx = pd.MultiIndex.from_frame(selected[["database", "gene_set"]])
        idx = idx.intersection(mu_all.index)

        if len(idx) < 5:
            continue

        mu_k = mu_all.loc[idx]
        w_k = w_all.loc[idx]

        M_k = mu_k.abs().mean(axis=0)
        U_k = np.log(w_k + DELTA).mean(axis=0)

        for lam in LAMBDA_GRID:
            score = M_k - lam * U_k
            res = survival_logrank(score, surv, label=f"K={K}, lambda={lam}", out_png=None)
            res["K"] = K
            res["lambda"] = lam
            res["n_gene_sets_used"] = len(idx)
            k_records.append(res)

    k_df = pd.DataFrame(k_records)
    k_df.to_csv(OUT_ROOT / f"{CANCER}_K_scale_D_W_PHY_survival.csv", index=False)

    if not k_df.empty:
        plt.figure(figsize=(6.5, 4.5))
        for lam in sorted(k_df["lambda"].dropna().unique()):
            sub = k_df[k_df["lambda"] == lam]
            plt.plot(sub["K"], sub["D_clinical"], marker="o", label=f"lambda={lam}")

        plt.xlabel("K selected gene sets")
        plt.ylabel("Clinical discriminability: -log10(p)")
        plt.title(f"{CANCER}: K-scale observation lens")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_ROOT / f"{CANCER}_K_scale_observation_lens.png", dpi=DPI)
        plt.close()

    # ------------------------------------------------------------
    # Gene-set-level clinical D for μ and W
    # ------------------------------------------------------------

    log("Computing gene-set-level clinical D for μ and W")

    gs_records = []

    for db, gs in mu_all.index:
        mu_score = mu_all.loc[(db, gs)]
        w_score = w_all.loc[(db, gs)]

        res_mu = survival_logrank(mu_score, surv, label=f"{db}:{gs}:mu", out_png=None)
        res_w = survival_logrank(w_score, surv, label=f"{db}:{gs}:W", out_png=None)

        gs_records.append({
            "database": db,
            "gene_set": gs,
            "D_mu": res_mu.get("D_clinical", np.nan),
            "p_mu": res_mu.get("p", np.nan),
            "D_W": res_w.get("D_clinical", np.nan),
            "p_W": res_w.get("p", np.nan),
            "n": res_mu.get("n", np.nan),
            "events": res_mu.get("events", np.nan),
        })

    gs_d = pd.DataFrame(gs_records)
    gs_d = gs_d.merge(summary_all, on=["database", "gene_set"], how="left")
    gs_d.to_csv(OUT_ROOT / f"{CANCER}_gene_set_level_mu_W_clinical_D.csv", index=False)

    top_mu = gs_d.sort_values("D_mu", ascending=False).head(10)
    top_w = gs_d.sort_values("D_W", ascending=False).head(10)

    top_dir = OUT_ROOT / "Top_KM_gene_sets"
    top_dir.mkdir(exist_ok=True)

    for _, row in top_mu.iterrows():
        db = row["database"]
        gs = row["gene_set"]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{db}_{gs}")[:180]

        survival_logrank(
            mu_all.loc[(db, gs)],
            surv,
            label=f"{db}:{gs} mu",
            out_png=top_dir / f"{CANCER}_TOP_mu_{safe_name}.png"
        )

    for _, row in top_w.iterrows():
        db = row["database"]
        gs = row["gene_set"]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{db}_{gs}")[:180]

        survival_logrank(
            w_all.loc[(db, gs)],
            surv,
            label=f"{db}:{gs} W",
            out_png=top_dir / f"{CANCER}_TOP_W_{safe_name}.png"
        )

    # ------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------

    summary_txt = OUT_ROOT / f"{CANCER}_RUN_SUMMARY.txt"

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"AIDO-D Plan B Part 2 — {CANCER}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"DATA_ROOT: {DATA_ROOT}\n")
        f.write(f"Expression file: {EXPR_FILE}\n")
        f.write(f"Survival file: {SURV_FILE}\n")
        f.write(f"Matched patients: {len(common)}\n")
        f.write(f"Retained gene sets: {summary_all.shape[0]}\n")
        f.write(f"Output root: {OUT_ROOT}\n\n")

        f.write("Processed GMT databases:\n")
        for db in summary_all["database"].unique():
            f.write(f"  - {db}: {(summary_all['database'] == db).sum()} gene sets\n")

        f.write("\nD_W-PHY lambda scan:\n")
        f.write(d_results_df.to_string(index=False))
        f.write("\n\n")

        if not k_df.empty:
            f.write("Top K-scale results:\n")
            f.write(k_df.sort_values("D_clinical", ascending=False).head(10).to_string(index=False))
            f.write("\n\n")

        f.write("Top μ gene sets:\n")
        f.write(
            top_mu[
                ["database", "gene_set", "D_mu", "p_mu", "n_genes_mapped", "W_mean", "coherence_mean"]
            ].to_string(index=False)
        )

        f.write("\n\nTop W gene sets:\n")
        f.write(
            top_w[
                ["database", "gene_set", "D_W", "p_W", "n_genes_mapped", "W_mean", "coherence_mean"]
            ].to_string(index=False)
        )

    log("Completed.")
    log(f"All outputs saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
