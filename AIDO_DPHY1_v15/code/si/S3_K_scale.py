#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AIDO-D-PHY-I SI V4
Supplementary Figure S3 V2 STRICT
Full K-scale observation-lens analysis across cancers and pathway databases

Purpose:
Fix the K*=0 problem by:
1. Excluding lambda-like and non-K-scale files.
2. Rejecting any candidate table whose K values round to 0.
3. Using stricter K-column detection.
4. Keeping only biologically/analytically plausible integer K values.

Panels:
A. Database-level median D_clinical(K) curves
B. Distribution of optimal K*
C. Maximum discriminability after K-scale optimization
D. K* versus achieved maximum D

Important:
K = number of selected BP/pathway observables in a patient-level score.
K is distinct from N_B, the number of mapped genes within one observable.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# User settings
# =========================================================

ROOT = Path(r"D:/AIDO-Temp")
OUTDIR = ROOT / "AIDO-D-PHY-I-SI-V4-FIGURES"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TABLE = OUTDIR / "SI_V4_FigureS3_V2_STRICT_K_scale_integrated_table.csv"
OUTPUT_OPTIMAL = OUTDIR / "SI_V4_FigureS3_V2_STRICT_K_scale_optimal_summary.csv"
OUTPUT_PNG = OUTDIR / "SI_V4_FigureS3_V2_STRICT_K_scale_observation_lens.png"
OUTPUT_PDF = OUTDIR / "SI_V4_FigureS3_V2_STRICT_K_scale_observation_lens.pdf"
DIAG_FILE = OUTDIR / "SI_V4_FigureS3_V2_STRICT_K_scale_diagnostics.csv"

CANCER_ORDER = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LAML", "LIHC",
    "LUAD", "LUSC", "OV", "PAAD", "PRAD", "STAD", "THCA", "UCEC"
]

DATABASE_ORDER = [
    "Hallmark", "GO:BP", "Reactome", "KEGG",
    "WikiPathways", "C2:CP", "BioCarta", "PID"
]

# Plausible K values. Adjust if your K-scale experiment used other values.
VALID_K_VALUES = {
    1, 2, 3, 4, 5,
    10, 15, 20, 25, 30, 40, 50,
    75, 100, 150, 200, 250, 300,
    400, 500, 750, 1000
}

# Folder filters.
INCLUDE_PATH_KEYWORDS = [
    "PlanB",
    "MeanFWHM",
    "D-PHY",
    "REAL_DATA",
    "PHY-I",
]

# File/path must contain at least one real K-scale signal.
# This is stricter than V1.
K_PATH_KEYWORDS = [
    "k_scale",
    "kscale",
    "k-scale",
    "observation_lens",
    "observation-lens",
    "lens",
    "selected_observables",
    "selected_process",
    "topK",
    "top_k",
]

# Explicit exclusions to avoid lambda tables and other contaminated sources.
EXCLUDE_PATH_KEYWORDS = [
    "RANDOM-TEST",
    "RandomTest",
    "MASTER-INTEGRATED",
    "FIGURE4_REVISED",
    "QC-AUDIT",
    "SF-FIGURES",
    "SF-EXTRACTED",
    "MANUSCRIPT",
    "lambda",
    "lam_",
    "lambda_scan",
    "D_W",
    "DW",
    "width_penalty",
    "penalty",
    "gain",
    "improvement",
]

MIN_ROWS = 2


# =========================================================
# Helper functions
# =========================================================

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def match_any(text, keywords):
    text = str(text).lower()
    return any(k.lower() in text for k in keywords)


def safe_read_table(path: Path):
    try:
        if path.suffix.lower() in [".tsv", ".txt"]:
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return None


def infer_cancer_from_path(path: Path):
    s = str(path).replace("\\", "/").upper()
    for c in CANCER_ORDER + ["READ", "COADREAD", "LGG", "SKCM", "SARC", "CESC", "ESCA"]:
        if re.search(rf"(^|[^A-Z0-9]){c}([^A-Z0-9]|$)", s):
            return c
    return "UNKNOWN"


def infer_database_from_text(text):
    t = str(text).upper()
    if "HALLMARK" in t:
        return "Hallmark"
    if "GO_BP" in t or "GOBP" in t or "GO-BP" in t or "BIOLOGICAL_PROCESS" in t:
        return "GO:BP"
    if "REACTOME" in t:
        return "Reactome"
    if "KEGG" in t:
        return "KEGG"
    if "WIKIPATHWAY" in t:
        return "WikiPathways"
    if "BIOCARTA" in t:
        return "BioCarta"
    if "PID" in t:
        return "PID"
    if "C2CP" in t or "C2_CP" in t or "CP_" in t:
        return "C2:CP"
    return "Unknown"


def pick_col_exact_first(columns, exact_candidates, regex_candidates=None):
    """
    Pick a column conservatively.
    exact_candidates are normalized exact names.
    regex_candidates are fallback patterns.
    """
    regex_candidates = regex_candidates or []
    norm_map = {norm_col(c): c for c in columns}
    keys = list(norm_map.keys())

    for cand in exact_candidates:
        c = norm_col(cand)
        if c in norm_map:
            return norm_map[c]

    for pattern in regex_candidates:
        for k in keys:
            if re.fullmatch(pattern, k, flags=re.I):
                return norm_map[k]

    for pattern in regex_candidates:
        for k in keys:
            if re.search(pattern, k, flags=re.I):
                return norm_map[k]

    return None


def should_scan_file(path: Path):
    s = str(path).replace("\\", "/")
    fname = path.name

    if not match_any(s, INCLUDE_PATH_KEYWORDS):
        return False
    if not match_any(s, K_PATH_KEYWORDS):
        return False
    if match_any(s, EXCLUDE_PATH_KEYWORDS):
        return False
    if path.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        return False

    # Avoid generic summaries if not explicitly K-scale.
    if match_any(fname, ["database_summary", "gene_set_summary", "max_D_summary"]):
        return False

    return True


def standardize_k_table(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        return None, "empty"

    cols = list(df.columns)

    # Conservative K detection.
    # Do NOT use broad candidates such as "selected" alone, because that can pick wrong columns.
    k_col = pick_col_exact_first(
        cols,
        exact_candidates=[
            "K", "k", "top_k", "topK",
            "K_value", "k_value",
            "n_selected_observables",
            "num_selected_observables",
            "selected_observable_count",
            "num_observables_selected",
            "n_process_observables",
            "num_process_observables"
        ],
        regex_candidates=[
            r"^k$",
            r"^top_?k$",
            r"^k_?value$",
            r"^n_?selected_?observables$",
            r"^num_?selected_?observables$",
            r"^selected_?observable_?count$",
            r"^num_?observables_?selected$"
        ]
    )

    d_col = pick_col_exact_first(
        cols,
        exact_candidates=[
            "D_clinical", "clinical_D", "Dclinical",
            "D", "max_D", "best_D", "D_max",
            "neg_log10_p", "minus_log10_p",
            "logrank_D", "survival_D", "discriminability"
        ],
        regex_candidates=[
            r"^d$",
            r".*clinical.*d.*",
            r".*max.*d.*",
            r".*log.*rank.*",
            r".*discrimin.*",
            r".*neg.*log.*10.*p.*"
        ]
    )

    if k_col is None:
        return None, "no_K_column"
    if d_col is None:
        return None, "no_D_column"

    database_col = pick_col_exact_first(
        cols,
        exact_candidates=["database", "db", "source", "collection", "gene_set_database", "pathway_database"],
        regex_candidates=[r".*database.*", r"^db$", r"^source$"]
    )

    cancer_col = pick_col_exact_first(
        cols,
        exact_candidates=["cancer", "cohort", "cancer_type", "tumor", "dataset"],
        regex_candidates=[r"^cancer.*", r"^cohort$", r"^dataset$"]
    )

    selection_col = pick_col_exact_first(
        cols,
        exact_candidates=["selection_rule", "rule", "method", "ranking", "score_type", "selection_method"],
        regex_candidates=[r".*selection.*rule.*", r"^method$", r"^ranking$"]
    )

    out = pd.DataFrame()
    out["source_file"] = [str(path)] * len(df)

    if cancer_col is not None:
        out["cancer"] = df[cancer_col].astype(str)
    else:
        out["cancer"] = infer_cancer_from_path(path)

    if database_col is not None:
        out["database"] = df[database_col].astype(str)
        out["database"] = out["database"].apply(
            lambda x: infer_database_from_text(x) if infer_database_from_text(x) != "Unknown" else str(x)
        )
    else:
        out["database"] = infer_database_from_text(path)

    if selection_col is not None:
        out["selection_rule"] = df[selection_col].astype(str)
    else:
        out["selection_rule"] = "unspecified"

    # Parse K and D.
    out["K_raw"] = pd.to_numeric(df[k_col], errors="coerce")
    out["D"] = pd.to_numeric(df[d_col], errors="coerce")

    # Clean cancer/database.
    bad_db = out["database"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_db, "database"] = infer_database_from_text(path)

    bad_cancer = out["cancer"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_cancer, "cancer"] = infer_cancer_from_path(path)

    for col in ["K_raw", "D"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=["K_raw", "D"])

    # CRITICAL FIX:
    # Round first, then remove K < 1.
    # V1 filtered K > 0 before rounding, so lambda-like values 0.1/0.2 could become K=0.
    out["K"] = out["K_raw"].round().astype(int)
    out = out[(out["K"] >= 1) & (out["D"] >= 0)]

    # Strict plausible K filter.
    out = out[out["K"].isin(VALID_K_VALUES)]

    # Remove cases that are almost certainly not K-scale.
    # If all K values are 1 only and there are no other K levels, it cannot support a K-scale curve.
    if out["K"].nunique() < 2:
        return None, f"too_few_K_levels_after_filter; K_values={sorted(out['K'].unique())}"

    out = out[out["cancer"].isin(CANCER_ORDER)]
    out = out[out["database"].isin(DATABASE_ORDER)]

    if len(out) < MIN_ROWS:
        return None, "too_few_rows_after_filter"

    out = out[["source_file", "cancer", "database", "selection_rule", "K", "D"]]

    return out, f"USED; k_col={k_col}; d_col={d_col}"


def collect_k_tables():
    diagnostics = []
    tables = []

    files = []
    for ext in ["*.csv", "*.tsv", "*.txt"]:
        files.extend(ROOT.rglob(ext))

    candidates = [p for p in files if should_scan_file(p)]

    print(f"[INFO] STRICT scan found {len(candidates)} candidate K-scale files.")

    for p in candidates:
        df = safe_read_table(p)
        if df is None or df.empty:
            diagnostics.append({
                "file": str(p),
                "status": "read_failed_or_empty",
                "n_rows": 0,
                "used_rows": 0,
                "K_values": "",
                "columns": ""
            })
            continue

        std, status = standardize_k_table(df, p)

        if std is not None and len(std) > 0:
            tables.append(std)
            diagnostics.append({
                "file": str(p),
                "status": status,
                "n_rows": len(df),
                "used_rows": len(std),
                "cancers": ";".join(sorted(std["cancer"].unique())),
                "databases": ";".join(sorted(std["database"].unique())),
                "K_min": std["K"].min(),
                "K_max": std["K"].max(),
                "K_values": ";".join(map(str, sorted(std["K"].unique()))),
                "D_max": std["D"].max(),
                "columns": " | ".join(map(str, df.columns))
            })
            print(
                f"[USED] {p} | rows={len(std)} | "
                f"cancer={sorted(std['cancer'].unique())} | "
                f"db={sorted(std['database'].unique())} | "
                f"K={sorted(std['K'].unique())}"
            )
        else:
            diagnostics.append({
                "file": str(p),
                "status": status,
                "n_rows": len(df),
                "used_rows": 0,
                "K_values": "",
                "columns": " | ".join(map(str, df.columns))
            })

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_FILE, index=False, encoding="utf-8-sig")

    if not tables:
        return pd.DataFrame()

    out = pd.concat(tables, ignore_index=True)
    out = out.drop_duplicates(subset=["cancer", "database", "selection_rule", "K", "D"])

    # Final hard check.
    out = out[out["K"] >= 1]

    return out


def make_optimal_summary(df: pd.DataFrame):
    """
    Aggregate to cancer/database/K using max D,
    then find K* and max D for each cancer/database.
    """
    curve = (
        df.groupby(["cancer", "database", "K"], as_index=False)["D"]
        .max()
        .sort_values(["cancer", "database", "K"])
    )

    # Safety check: no K=0 should survive.
    curve = curve[curve["K"] >= 1]

    idx = curve.groupby(["cancer", "database"])["D"].idxmax()
    optimal = curve.loc[idx].copy()
    optimal = optimal.rename(columns={"K": "K_star", "D": "max_D"})
    optimal = optimal.sort_values(["cancer", "database"])

    optimal = optimal[optimal["K_star"] >= 1]

    return curve, optimal


# =========================================================
# Figure drawing
# =========================================================

def make_figure(curve: pd.DataFrame, optimal: pd.DataFrame):
    cancer_order = [c for c in CANCER_ORDER if c in set(curve["cancer"])]
    db_order = [d for d in DATABASE_ORDER if d in set(curve["database"])]

    fig = plt.figure(figsize=(15.5, 12.5))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.30)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    # -----------------------------------------------------
    # Panel A: database-level median D(K) curves
    # -----------------------------------------------------
    db_curve = (
        curve.groupby(["database", "K"], as_index=False)["D"]
        .median()
        .sort_values(["database", "K"])
    )

    for i, db in enumerate(db_order):
        sub = db_curve[db_curve["database"] == db].sort_values("K")
        if sub.empty:
            continue

        axA.plot(
            sub["K"],
            sub["D"],
            marker=markers[i % len(markers)],
            linewidth=2.2,
            markersize=4.0,
            label=db
        )

    if curve["K"].max() / max(curve["K"].min(), 1) > 20:
        axA.set_xscale("log")

    axA.set_xlabel(r"Number of selected process-level observables, $K$")
    axA.set_ylabel(r"Clinical discriminability, $D_{\mathrm{clinical}}$")
    axA.set_title(r"A | Database-level median $D_{\mathrm{clinical}}(K)$ curves",
                  loc="left", fontweight="bold")
    axA.grid(True, alpha=0.25)
    axA.legend(frameon=False, fontsize=9)

    # -----------------------------------------------------
    # Panel B: distribution of K*
    # -----------------------------------------------------
    k_values = optimal["K_star"].dropna()

    if len(k_values) > 0:
        bins = min(12, max(4, k_values.nunique()))
        axB.hist(k_values, bins=bins)
    else:
        axB.text(0.5, 0.5, "No K* values", ha="center", va="center", transform=axB.transAxes)

    if len(k_values) > 0 and (k_values.max() / max(k_values.min(), 1) > 20):
        axB.set_xscale("log")

    axB.set_xlabel(r"Optimal observation-lens scale, $K^*$")
    axB.set_ylabel("Number of cancer/database pairs")
    axB.set_title(r"B | Distribution of optimal $K^*$",
                  loc="left", fontweight="bold")
    axB.grid(True, alpha=0.25)

    # -----------------------------------------------------
    # Panel C: maximum D after K-scale optimization
    # -----------------------------------------------------
    pivot_maxd = (
        optimal.pivot(index="cancer", columns="database", values="max_D")
        .reindex(index=cancer_order, columns=db_order)
    )

    im = axC.imshow(pivot_maxd.values, aspect="auto")
    axC.set_xticks(np.arange(len(pivot_maxd.columns)))
    axC.set_xticklabels(pivot_maxd.columns, rotation=45, ha="right")
    axC.set_yticks(np.arange(len(pivot_maxd.index)))
    axC.set_yticklabels(pivot_maxd.index)
    axC.set_title(r"C | Maximum discriminability after K-scale optimization",
                  loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.set_label(r"Maximum $D_{\mathrm{clinical}}$")

    # -----------------------------------------------------
    # Panel D: K* vs achieved maximum D
    # -----------------------------------------------------
    for i, db in enumerate(db_order):
        sub = optimal[optimal["database"] == db]
        if sub.empty:
            continue

        axD.scatter(
            sub["K_star"],
            sub["max_D"],
            s=70,
            alpha=0.75,
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.4,
            label=db
        )

    for _, row in optimal.iterrows():
        axD.text(
            row["K_star"],
            row["max_D"],
            row["cancer"],
            fontsize=6.5,
            alpha=0.75
        )

    if optimal["K_star"].max() / max(optimal["K_star"].min(), 1) > 20:
        axD.set_xscale("log")

    axD.set_xlabel(r"Optimal observation-lens scale, $K^*$")
    axD.set_ylabel(r"Achieved maximum $D_{\mathrm{clinical}}$")
    axD.set_title(r"D | $K^*$ versus achieved maximum D",
                  loc="left", fontweight="bold")
    axD.grid(True, alpha=0.25)
    axD.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Supplementary Figure S3. Full K-scale observation-lens analysis across cancers and pathway databases",
        fontsize=15, fontweight="bold", y=0.985
    )

    fig.text(
        0.5, 0.012,
        r"$K$ denotes the number of selected BP/pathway observables in a patient-level score; it is distinct from $N_B$, the number of genes within one observable.",
        ha="center", fontsize=10
    )

    fig.savefig(OUTPUT_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = collect_k_tables()

    if df.empty:
        print("\n[ERROR] No usable STRICT K-scale tables were found.")
        print(f"[INFO] Diagnostic file saved to: {DIAG_FILE}")
        print("\nPlease check diagnostics. Need files/columns similar to:")
        print("K or top_k or n_selected_observables; D_clinical or max_D; cancer; database")
        return

    df.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")

    curve, optimal = make_optimal_summary(df)
    optimal.to_csv(OUTPUT_OPTIMAL, index=False, encoding="utf-8-sig")

    print("\n[INFO] STRICT integrated K-scale table saved:")
    print(f"       {OUTPUT_TABLE}")
    print(f"[INFO] STRICT optimal summary saved:")
    print(f"       {OUTPUT_OPTIMAL}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Cancers: {sorted(df['cancer'].unique())}")
    print(f"[INFO] Databases: {sorted(df['database'].unique())}")
    print(f"[INFO] K range: {df['K'].min()} - {df['K'].max()}")
    print(f"[INFO] K values: {sorted(df['K'].unique())}")

    if (optimal["K_star"] <= 0).any():
        raise ValueError("K_star <= 0 detected. Strict filtering failed.")

    make_figure(curve, optimal)

    print("\n[DONE] Supplementary Figure S3 V2 STRICT generated:")
    print(f"       PNG: {OUTPUT_PNG}")
    print(f"       PDF: {OUTPUT_PDF}")
    print(f"       Diagnostics: {DIAG_FILE}")


if __name__ == "__main__":
    main()
