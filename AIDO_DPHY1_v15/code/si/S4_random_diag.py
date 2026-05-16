#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AIDO-D-PHY-I SI V4
Supplementary Figure S4 (V2)
Random-background diagnostics and biological-overlap analysis across cancers

Panels:
A. Structured BP fraction above random 95th percentile
B. Structured max D vs random max D
C. Structured advantage over random max D (Delta D)
D. Maximum Jaccard overlap of high-D random sets

Purpose:
Support Supplementary Note S4 and align with main Figure 4 logic.

Panel C in this version is:
Delta D = structured_max_D - random_max_D

This avoids leaving an empty panel when D + coherence summary is unavailable.
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

OUTPUT_TABLE = OUTDIR / "SI_V4_FigureS4_V2_random_background_integrated_table.csv"
OUTPUT_PNG = OUTDIR / "SI_V4_FigureS4_V2_random_background_diagnostics.png"
OUTPUT_PDF = OUTDIR / "SI_V4_FigureS4_V2_random_background_diagnostics.pdf"
DIAG_FILE = OUTDIR / "SI_V4_FigureS4_V2_random_background_diagnostics_scan.csv"

CANCER_ORDER = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LAML", "LIHC",
    "LUAD", "LUSC", "OV", "PAAD", "PRAD", "STAD", "THCA", "UCEC"
]

DATABASE_ORDER = [
    "Hallmark", "GO:BP", "Reactome", "KEGG", "WikiPathways", "C2:CP", "BioCarta", "PID"
]

INCLUDE_PATH_KEYWORDS = [
    "RANDOM-TEST",
    "RandomTest",
    "MASTER-INTEGRATED",
    "FIGURE4_REVISED",
    "figure4",
    "random_vs_structured",
]

EXCLUDE_PATH_KEYWORDS = [
    "lambda",
    "K_scale",
    "observation_lens",
    "MeanFWHM_EXP-W0",
]

MIN_METRIC_COUNT = 1


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


def pick_col(columns, exact_candidates=None, regex_candidates=None):
    exact_candidates = exact_candidates or []
    regex_candidates = regex_candidates or []
    norm_map = {norm_col(c): c for c in columns}
    keys = list(norm_map.keys())

    for cand in exact_candidates:
        c = norm_col(cand)
        if c in norm_map:
            return norm_map[c]

    for pat in regex_candidates:
        for k in keys:
            if re.fullmatch(pat, k, flags=re.I):
                return norm_map[k]

    for pat in regex_candidates:
        for k in keys:
            if re.search(pat, k, flags=re.I):
                return norm_map[k]

    return None


def should_scan_file(path: Path):
    s = str(path).replace("\\", "/")
    if not match_any(s, INCLUDE_PATH_KEYWORDS):
        return False
    if match_any(s, EXCLUDE_PATH_KEYWORDS):
        return False
    if path.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        return False
    return True


def standardize_random_summary(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        return None, "empty"

    cols = list(df.columns)

    # identity columns
    database_col = pick_col(
        cols,
        exact_candidates=["database", "db", "source", "collection", "gene_set_database", "pathway_database"],
        regex_candidates=[r".*database.*", r"^db$", r"^source$"]
    )

    cancer_col = pick_col(
        cols,
        exact_candidates=["cancer", "cohort", "cancer_type", "dataset", "tumor"],
        regex_candidates=[r"^cancer.*", r"^cohort$", r"^dataset$"]
    )

    # metrics
    frac_above95_col = pick_col(
        cols,
        exact_candidates=[
            "structured_above_random95_fraction",
            "fraction_structured_above_random95",
            "fraction_above_random95",
            "panelA_fraction",
            "frac_above_random95"
        ],
        regex_candidates=[
            r".*above.*random.*95.*fraction.*",
            r".*fraction.*above.*95.*",
            r".*structured.*95.*percentile.*fraction.*",
            r".*panel.*a.*fraction.*"
        ]
    )

    structured_max_d_col = pick_col(
        cols,
        exact_candidates=["structured_max_D", "max_D_structured", "structured_best_D"],
        regex_candidates=[r".*structured.*max.*d.*", r".*max.*d.*structured.*"]
    )

    random_max_d_col = pick_col(
        cols,
        exact_candidates=["random_max_D", "max_D_random", "random_best_D"],
        regex_candidates=[r".*random.*max.*d.*", r".*max.*d.*random.*"]
    )

    max_jaccard_col = pick_col(
        cols,
        exact_candidates=["max_jaccard_overlap", "maximum_jaccard_overlap", "jaccard_max", "panelD_jaccard"],
        regex_candidates=[
            r".*max.*jaccard.*",
            r".*jaccard.*max.*",
            r".*maximum.*jaccard.*",
            r".*panel.*d.*jaccard.*"
        ]
    )

    metric_cols = {
        "fraction_above_random95": frac_above95_col,
        "structured_max_D": structured_max_d_col,
        "random_max_D": random_max_d_col,
        "max_jaccard_overlap": max_jaccard_col,
    }

    detected = {k: v for k, v in metric_cols.items() if v is not None}
    if len(detected) < MIN_METRIC_COUNT:
        return None, "no_supported_metric_columns"

    out = pd.DataFrame(index=df.index)
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

    bad_db = out["database"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_db, "database"] = infer_database_from_text(path)

    bad_cancer = out["cancer"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_cancer, "cancer"] = infer_cancer_from_path(path)

    for metric_name, col in metric_cols.items():
        if col is not None:
            out[metric_name] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[metric_name] = np.nan

    for col in ["fraction_above_random95", "structured_max_D", "random_max_D", "max_jaccard_overlap"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    out = out[out["cancer"].isin(CANCER_ORDER)]
    out = out[out["database"].isin(DATABASE_ORDER)]

    # clip valid ranges
    if "fraction_above_random95" in out.columns:
        out.loc[out["fraction_above_random95"] < 0, "fraction_above_random95"] = np.nan
        out.loc[out["fraction_above_random95"] > 1.0, "fraction_above_random95"] = np.nan

    if "max_jaccard_overlap" in out.columns:
        out.loc[out["max_jaccard_overlap"] < 0, "max_jaccard_overlap"] = np.nan
        out.loc[out["max_jaccard_overlap"] > 1.0, "max_jaccard_overlap"] = np.nan

    for col in ["structured_max_D", "random_max_D"]:
        if col in out.columns:
            out.loc[out[col] < 0, col] = np.nan

    metric_present = out[
        ["fraction_above_random95", "structured_max_D", "random_max_D", "max_jaccard_overlap"]
    ].notna().any(axis=1)
    out = out[metric_present]

    if out.empty:
        return None, "all_rows_filtered_out"

    return out, f"USED; metrics={','.join(detected.keys())}"


def collect_random_summaries():
    diagnostics = []
    tables = []

    files = []
    for ext in ["*.csv", "*.tsv", "*.txt"]:
        files.extend(ROOT.rglob(ext))

    candidates = [p for p in files if should_scan_file(p)]

    print(f"[INFO] Scan found {len(candidates)} candidate random-background files.")

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

        std, status = standardize_random_summary(df, p)

        if std is not None and len(std) > 0:
            tables.append(std)
            diagnostics.append({
                "file": str(p),
                "status": status,
                "n_rows": len(df),
                "used_rows": len(std),
                "cancers": ";".join(sorted(std["cancer"].unique())),
                "databases": ";".join(sorted(std["database"].unique())),
                "columns": " | ".join(map(str, df.columns))
            })
            print(
                f"[USED] {p} | rows={len(std)} | "
                f"cancer={sorted(std['cancer'].unique())} | "
                f"db={sorted(std['database'].unique())}"
            )
        else:
            diagnostics.append({
                "file": str(p),
                "status": status,
                "n_rows": len(df),
                "used_rows": 0,
                "columns": " | ".join(map(str, df.columns))
            })

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_FILE, index=False, encoding="utf-8-sig")

    if not tables:
        return pd.DataFrame()

    out = pd.concat(tables, ignore_index=True)

    def agg_nonnull_max(x):
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return x.max()

    merged = (
        out.groupby(["cancer", "database"], as_index=False)
        .agg({
            "fraction_above_random95": agg_nonnull_max,
            "structured_max_D": agg_nonnull_max,
            "random_max_D": agg_nonnull_max,
            "max_jaccard_overlap": agg_nonnull_max
        })
    )

    merged["delta_D"] = merged["structured_max_D"] - merged["random_max_D"]

    return merged


# =========================================================
# Figure drawing
# =========================================================

def plot_heatmap(ax, pivot, title, cbar_label, fig):
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    return im


def make_figure(df: pd.DataFrame):
    cancer_order = [c for c in CANCER_ORDER if c in set(df["cancer"])]
    db_order = [d for d in DATABASE_ORDER if d in set(df["database"])]

    fig = plt.figure(figsize=(15.5, 12.5))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.30)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # -----------------------------------------------------
    # Panel A: fraction above random 95th
    # -----------------------------------------------------
    pivot_A = (
        df.pivot(index="cancer", columns="database", values="fraction_above_random95")
        .reindex(index=cancer_order, columns=db_order)
    )
    plot_heatmap(
        axA, pivot_A,
        r"A | Structured BP fraction above random 95th percentile",
        r"Fraction above random 95th percentile", fig
    )

    # -----------------------------------------------------
    # Panel B: structured max D vs random max D
    # -----------------------------------------------------
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    plotted_any = False

    for i, db in enumerate(db_order):
        sub = df[df["database"] == db].dropna(subset=["structured_max_D", "random_max_D"])
        if sub.empty:
            continue
        plotted_any = True
        axB.scatter(
            sub["random_max_D"],
            sub["structured_max_D"],
            s=70,
            alpha=0.75,
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.4,
            label=db
        )

        # only label notable outliers to reduce clutter
        for _, row in sub.iterrows():
            if abs(row["structured_max_D"] - row["random_max_D"]) >= 2.0 or row["structured_max_D"] >= 8.0 or row["random_max_D"] >= 8.0:
                axB.text(row["random_max_D"], row["structured_max_D"], row["cancer"], fontsize=6.5, alpha=0.8)

    if plotted_any:
        all_x = df["random_max_D"].dropna()
        all_y = df["structured_max_D"].dropna()
        if len(all_x) > 0 and len(all_y) > 0:
            lo = min(all_x.min(), all_y.min())
            hi = max(all_x.max(), all_y.max())
            axB.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        axB.legend(frameon=False, fontsize=8)

    axB.set_xlabel(r"Random maximum $D_{\mathrm{clinical}}$")
    axB.set_ylabel(r"Structured maximum $D_{\mathrm{clinical}}$")
    axB.set_title(r"B | Structured max D versus random max D", loc="left", fontweight="bold")
    axB.grid(True, alpha=0.25)

    # -----------------------------------------------------
    # Panel C: Delta D = structured max D - random max D
    # -----------------------------------------------------
    pivot_C = (
        df.pivot(index="cancer", columns="database", values="delta_D")
        .reindex(index=cancer_order, columns=db_order)
    )
    plot_heatmap(
        axC, pivot_C,
        r"C | Structured advantage over random max D",
        r"$\Delta D = D_{\mathrm{structured,max}} - D_{\mathrm{random,max}}$", fig
    )

    # -----------------------------------------------------
    # Panel D: max Jaccard overlap
    # -----------------------------------------------------
    pivot_D = (
        df.pivot(index="cancer", columns="database", values="max_jaccard_overlap")
        .reindex(index=cancer_order, columns=db_order)
    )
    plot_heatmap(
        axD, pivot_D,
        r"D | Maximum Jaccard overlap of high-D random sets",
        r"Maximum Jaccard overlap", fig
    )

    fig.suptitle(
        "Supplementary Figure S4. Random-background diagnostics and biological-overlap analysis across cancers",
        fontsize=15, fontweight="bold", y=0.985
    )

    fig.text(
        0.5, 0.012,
        r"Structured observables were evaluated against random-background expectation, structured-versus-random maximum discriminability, and biological-overlap diagnostics.",
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

    df = collect_random_summaries()

    if df.empty:
        print("\n[ERROR] No usable random-background summary tables were found.")
        print(f"[INFO] Diagnostic file saved to: {DIAG_FILE}")
        print("\nNeed summary tables with columns similar to:")
        print("fraction_above_random95, structured_max_D, random_max_D, max_jaccard_overlap")
        return

    df.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")

    print("\n[INFO] Integrated random-background table saved:")
    print(f"       {OUTPUT_TABLE}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Cancers: {sorted(df['cancer'].unique())}")
    print(f"[INFO] Databases: {sorted(df['database'].unique())}")

    make_figure(df)

    print("\n[DONE] Supplementary Figure S4 V2 generated:")
    print(f"       PNG: {OUTPUT_PNG}")
    print(f"       PDF: {OUTPUT_PDF}")
    print(f"       Diagnostics: {DIAG_FILE}")


if __name__ == "__main__":
    main()