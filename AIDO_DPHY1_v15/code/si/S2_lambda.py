#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AIDO-D-PHY-I SI V4
Supplementary Figure S2 (V2)
Full lambda-scan analysis across cancers and pathway databases

Panels:
A. Database-level median D_clinical(lambda) curves
B. Distribution of optimal lambda*
C. Maximum discriminability after lambda-scan
D. lambda* versus achieved maximum D
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

OUTPUT_TABLE = OUTDIR / "SI_V4_FigureS2_lambda_scan_integrated_table.csv"
OUTPUT_OPTIMAL = OUTDIR / "SI_V4_FigureS2_lambda_scan_optimal_summary.csv"
OUTPUT_PNG = OUTDIR / "SI_V4_FigureS2_lambda_scan_V2.png"
OUTPUT_PDF = OUTDIR / "SI_V4_FigureS2_lambda_scan_V2.pdf"
DIAG_FILE = OUTDIR / "SI_V4_FigureS2_lambda_scan_diagnostics.csv"

CANCER_ORDER = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LAML", "LIHC",
    "LUAD", "LUSC", "OV", "PAAD", "PRAD", "STAD", "THCA", "UCEC"
]

DATABASE_ORDER = [
    "Hallmark", "GO:BP", "Reactome", "KEGG",
    "WikiPathways", "C2:CP", "BioCarta", "PID"
]

INCLUDE_PATH_KEYWORDS = [
    "PlanB",
    "MeanFWHM",
    "D-PHY",
    "REAL_DATA",
]

LAMBDA_KEYWORDS = [
    "lambda",
    "lam",
    "DW",
    "D_W",
    "width_penalty",
    "penalty",
    "scan",
]

EXCLUDE_PATH_KEYWORDS = [
    "RANDOM-TEST",
    "RandomTest",
    "MASTER-INTEGRATED",
    "FIGURE4_REVISED",
    "QC-AUDIT",
    "SF-FIGURES",
    "SF-EXTRACTED",
    "MANUSCRIPT",
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


def pick_col(columns, candidates):
    norm_map = {norm_col(c): c for c in columns}
    keys = list(norm_map.keys())

    # exact match first
    for cand in candidates:
        c = norm_col(cand)
        if c in norm_map:
            return norm_map[c]

    # regex-like fallback
    for cand in candidates:
        for k in keys:
            if re.search(cand, k, flags=re.I):
                return norm_map[k]

    return None


def has_lambda_column(df):
    col = pick_col(df.columns, [
        "lambda", "lambda_value", "lambda_penalty", "lam", "lambda_star",
        r"^lambda$", r".*lambda.*", r"^lam$"
    ])
    return col is not None


def should_scan_file(path: Path):
    s = str(path).replace("\\", "/")
    if not match_any(s, INCLUDE_PATH_KEYWORDS):
        return False
    if match_any(s, EXCLUDE_PATH_KEYWORDS):
        return False
    if path.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        return False
    return True


def standardize_lambda_table(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        return None

    cols = list(df.columns)

    lambda_col = pick_col(cols, [
        "lambda", "lambda_value", "lambda_penalty", "lam",
        r"^lambda$", r".*lambda.*", r"^lam$"
    ])

    d_col = pick_col(cols, [
        "D_clinical", "clinical_D", "Dclinical",
        "D", "max_D", "best_D", "D_max",
        "neg_log10_p", "minus_log10_p", "logrank_D",
        "survival_D", "discriminability",
        r".*clinical.*d.*", r".*max.*d.*", r"^d$", r".*log.*rank.*", r".*discrimin.*"
    ])

    if lambda_col is None or d_col is None:
        return None

    database_col = pick_col(cols, [
        "database", "db", "source", "collection", "gene_set_database", "pathway_database"
    ])

    cancer_col = pick_col(cols, [
        "cancer", "cohort", "cancer_type", "tumor", "dataset"
    ])

    observable_col = pick_col(cols, [
        "observable", "gene_set", "geneset", "gene_set_name", "pathway",
        "pathway_name", "bp", "bp_name", "term", "term_name"
    ])

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

    if observable_col is not None:
        out["observable"] = df[observable_col].astype(str)
    else:
        out["observable"] = "unknown"

    out["lambda"] = pd.to_numeric(df[lambda_col], errors="coerce")
    out["D"] = pd.to_numeric(df[d_col], errors="coerce")

    # Clean cancer/database
    bad_db = out["database"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_db, "database"] = infer_database_from_text(path)

    bad_cancer = out["cancer"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_cancer, "cancer"] = infer_cancer_from_path(path)

    # replace infs
    for col in ["lambda", "D"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=["lambda", "D"])
    out = out[(out["D"] >= 0) & (out["lambda"] >= 0)]
    out = out[out["cancer"].isin(CANCER_ORDER)]
    out = out[out["database"].isin(DATABASE_ORDER)]

    if len(out) < MIN_ROWS:
        return None

    return out


def collect_lambda_tables():
    diagnostics = []
    tables = []

    files = []
    for ext in ["*.csv", "*.tsv", "*.txt"]:
        files.extend(ROOT.rglob(ext))

    candidates = [p for p in files if should_scan_file(p)]

    print(f"[INFO] Scanning {len(candidates)} candidate files for lambda-scan results.")

    for p in candidates:
        df = safe_read_table(p)
        if df is None or df.empty:
            diagnostics.append({
                "file": str(p),
                "status": "read_failed_or_empty",
                "n_rows": 0,
                "used_rows": 0,
                "columns": ""
            })
            continue

        filename_lambda_like = match_any(p.name, LAMBDA_KEYWORDS)
        if (not filename_lambda_like) and (not has_lambda_column(df)):
            diagnostics.append({
                "file": str(p),
                "status": "skipped_no_lambda_signal",
                "n_rows": len(df),
                "used_rows": 0,
                "columns": " | ".join(map(str, df.columns))
            })
            continue

        std = standardize_lambda_table(df, p)

        if std is not None and len(std) > 0:
            tables.append(std)
            diagnostics.append({
                "file": str(p),
                "status": "USED",
                "n_rows": len(df),
                "used_rows": len(std),
                "cancers": ";".join(sorted(std["cancer"].unique())),
                "databases": ";".join(sorted(std["database"].unique())),
                "lambda_min": std["lambda"].min(),
                "lambda_max": std["lambda"].max(),
                "D_max": std["D"].max(),
                "columns": " | ".join(map(str, df.columns))
            })
            print(f"[USED] {p} | rows={len(std)} | cancer={sorted(std['cancer'].unique())} | db={sorted(std['database'].unique())}")
        else:
            diagnostics.append({
                "file": str(p),
                "status": "not_usable_after_standardization",
                "n_rows": len(df),
                "used_rows": 0,
                "columns": " | ".join(map(str, df.columns))
            })

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_FILE, index=False, encoding="utf-8-sig")

    if not tables:
        return pd.DataFrame()

    out = pd.concat(tables, ignore_index=True)
    out = out.drop_duplicates(subset=["cancer", "database", "observable", "lambda", "D"])

    return out


def make_optimal_summary(df: pd.DataFrame):
    """
    Aggregate to cancer/database/lambda using max D,
    then find lambda* and max D for each cancer/database.
    """
    curve = (
        df.groupby(["cancer", "database", "lambda"], as_index=False)["D"]
        .max()
        .sort_values(["cancer", "database", "lambda"])
    )

    idx = curve.groupby(["cancer", "database"])["D"].idxmax()
    optimal = curve.loc[idx].copy()
    optimal = optimal.rename(columns={"lambda": "lambda_star", "D": "max_D"})
    optimal = optimal.sort_values(["cancer", "database"])

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

    # -----------------------------------------------------
    # Panel A: database-level median D(lambda) curves
    # -----------------------------------------------------
    db_curve = (
        curve.groupby(["database", "lambda"], as_index=False)["D"]
        .median()
        .sort_values(["database", "lambda"])
    )

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, db in enumerate(db_order):
        sub = db_curve[db_curve["database"] == db].sort_values("lambda")
        if sub.empty:
            continue

        axA.plot(
            sub["lambda"],
            sub["D"],
            marker=markers[i % len(markers)],
            linewidth=2.2,
            markersize=4.0,
            label=db
        )

    axA.set_xlabel(r"Width-penalty parameter, $\lambda$")
    axA.set_ylabel(r"Clinical discriminability, $D_{\mathrm{clinical}}$")
    axA.set_title(r"A | Database-level median $D_{\mathrm{clinical}}(\lambda)$ curves",
                  loc="left", fontweight="bold")
    axA.grid(True, alpha=0.25)
    axA.legend(frameon=False, fontsize=9)

    # -----------------------------------------------------
    # Panel B: distribution of lambda*
    # -----------------------------------------------------
    n_unique_lambda = optimal["lambda_star"].nunique()
    bins = min(12, max(4, n_unique_lambda))

    axB.hist(optimal["lambda_star"].dropna(), bins=bins)
    axB.set_xlabel(r"Optimal width-penalty parameter, $\lambda^*$")
    axB.set_ylabel("Number of cancer/database pairs")
    axB.set_title(r"B | Distribution of optimal $\lambda^*$",
                  loc="left", fontweight="bold")
    axB.grid(True, alpha=0.25)

    # -----------------------------------------------------
    # Panel C: maximum D after lambda-scan
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
    axC.set_title(r"C | Maximum discriminability after $\lambda$-scan",
                  loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.set_label(r"Maximum $D_{\mathrm{clinical}}$")

    # -----------------------------------------------------
    # Panel D: lambda* vs achieved maximum D
    # -----------------------------------------------------
    for i, db in enumerate(db_order):
        sub = optimal[optimal["database"] == db]
        if sub.empty:
            continue

        axD.scatter(
            sub["lambda_star"],
            sub["max_D"],
            s=70,
            alpha=0.75,
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.4,
            label=db
        )

    # cancer labels
    for _, row in optimal.iterrows():
        axD.text(
            row["lambda_star"],
            row["max_D"],
            row["cancer"],
            fontsize=6.5,
            alpha=0.75
        )

    axD.set_xlabel(r"Optimal width-penalty parameter, $\lambda^*$")
    axD.set_ylabel(r"Achieved maximum $D_{\mathrm{clinical}}$")
    axD.set_title(r"D | $\lambda^*$ versus achieved maximum D",
                  loc="left", fontweight="bold")
    axD.grid(True, alpha=0.25)
    axD.legend(frameon=False, fontsize=8)

    # -----------------------------------------------------
    # Figure title / footer
    # -----------------------------------------------------
    fig.suptitle(
        "Supplementary Figure S2. Full lambda-scan analysis across cancers and pathway databases",
        fontsize=15, fontweight="bold", y=0.985
    )

    fig.text(
        0.5, 0.012,
        r"$\lambda$ controls the width penalty and is interpreted as a sensitivity parameter, not a universal biological constant.",
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

    df = collect_lambda_tables()

    if df.empty:
        print("\n[ERROR] No usable lambda-scan tables were found.")
        print(f"[INFO] Diagnostic file saved to: {DIAG_FILE}")
        print("\nPlease check whether lambda-scan CSVs contain columns like:")
        print("cancer, database, lambda, D_clinical / max_D / discriminability / -log10p")
        return

    df.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")

    curve, optimal = make_optimal_summary(df)
    optimal.to_csv(OUTPUT_OPTIMAL, index=False, encoding="utf-8-sig")

    print("\n[INFO] Integrated lambda-scan table saved:")
    print(f"       {OUTPUT_TABLE}")
    print(f"[INFO] Optimal summary saved:")
    print(f"       {OUTPUT_OPTIMAL}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Cancers: {sorted(df['cancer'].unique())}")
    print(f"[INFO] Databases: {sorted(df['database'].unique())}")

    make_figure(curve, optimal)

    print("\n[DONE] Supplementary Figure S2 generated:")
    print(f"       PNG: {OUTPUT_PNG}")
    print(f"       PDF: {OUTPUT_PDF}")
    print(f"       Diagnostics: {DIAG_FILE}")


if __name__ == "__main__":
    main()