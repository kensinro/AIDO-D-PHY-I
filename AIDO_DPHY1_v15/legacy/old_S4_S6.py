# ============================================================
# AIDO-D-PHY Supplementary Figures 4–6 Builder
# Version: 2026-05-13
#
# Purpose:
#   Generate REAL-DATA supplementary figures from extracted CSV files.
#
# Input folder:
#   D:/AIDO-Temp/AIDO-D-PHY-SF-EXTRACTED
#
# Required / preferred files:
#   SF4_structured_vs_random_summary_all.csv
#   SF4_structured_D_all.csv
#   SF4_random_D_all.csv
#   SF5_pathway_size_vs_W.csv
#   SF3_K_scale_all_cancers.csv
#   SF6_K_lambda_heatmap_table.csv
#
# Output folder:
#   D:/AIDO-Temp/AIDO-D-PHY-SF-FIGURES
#
# Required packages:
#   pip install pandas numpy matplotlib
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. Paths and settings
# ============================================================

INPUT_DIR = Path(r"D:/AIDO-Temp/AIDO-D-PHY-SF-EXTRACTED")
OUT_DIR = Path(r"D:/AIDO-Temp/AIDO-D-PHY-SF-FIGURES")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

CANCER_ORDER = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC",
    "LAML", "LIHC", "LUAD", "LUSC", "OV", "PAAD",
    "PRAD", "SARC", "SKCM", "STAD", "THCA", "UCEC"
]

DATABASE_ORDER = ["Hallmark", "GO_BP", "Reactome", "KEGG_Medicus", "WikiPathways"]


# ============================================================
# 1. Utilities
# ============================================================

def log(msg):
    print(f"[AIDO-D-PHY SF FIG4-6] {msg}", flush=True)


def read_optional(filename):
    path = INPUT_DIR / filename
    if not path.exists():
        log(f"Optional file missing: {path}")
        return None
    df = pd.read_csv(path)
    log(f"Loaded {filename}: {df.shape}")
    return df


def read_required(filename):
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    log(f"Loaded {filename}: {df.shape}")
    return df


def order_cancers(values):
    values = set(str(v) for v in values)
    return [c for c in CANCER_ORDER if c in values] + sorted([c for c in values if c not in CANCER_ORDER])


def order_databases(values):
    values = set(str(v) for v in values)
    return [d for d in DATABASE_ORDER if d in values] + sorted([d for d in values if d not in DATABASE_ORDER])


def savefig(name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    log(f"Saved: {png}")
    log(f"Saved: {pdf}")


def panel(ax, label):
    ax.text(
        -0.10, 1.08, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )


def clean_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of candidate columns found: {candidates}. Available={list(df.columns)}")
    return None


# ============================================================
# 2. Supplementary Figure 4
# Structured-versus-random global distributions
# ============================================================

def make_sf4():
    """
    SF4:
      A. Structured vs random D distributions
      B. Structured mean D vs random mean D by cancer/database
      C. Fraction of structured pathways above random 95th percentile
      D. Random 95th percentile vs structured max D

    Inputs:
      SF4_structured_D_all.csv
      SF4_random_D_all.csv
      SF4_structured_vs_random_summary_all.csv
    """

    structured = read_optional("SF4_structured_D_all.csv")
    random_df = read_optional("SF4_random_D_all.csv")
    summary = read_optional("SF4_structured_vs_random_summary_all.csv")

    if structured is None or random_df is None:
        raise FileNotFoundError(
            "SF4 requires SF4_structured_D_all.csv and SF4_random_D_all.csv. "
            "Run extractor after random-test results are available."
        )

    d_col_s = get_col(structured, ["D", "D_clinical"])
    d_col_r = get_col(random_df, ["D", "D_clinical"])

    structured = clean_numeric(structured, [d_col_s])
    random_df = clean_numeric(random_df, [d_col_r])

    structured = structured.dropna(subset=[d_col_s])
    random_df = random_df.dropna(subset=[d_col_r])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)

    # A. distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(structured[d_col_s].values, bins=60, alpha=0.6, density=True, label="Structured")
    ax.hist(random_df[d_col_r].values, bins=60, alpha=0.6, density=True, label="Random")
    ax.set_xlabel("D = -log10(p)")
    ax.set_ylabel("Density")
    ax.set_title("Global structured vs random D distributions")
    ax.legend(frameon=False)
    panel(ax, "A")

    # B/C/D use summary if available; otherwise compute minimal summaries
    if summary is None or summary.empty:
        log("Summary file missing; computing basic cancer/database summaries from raw D files.")
        s_cancer_col = get_col(structured, ["Cancer"], required=False)
        s_db_col = get_col(structured, ["database", "Database"], required=False)
        r_cancer_col = get_col(random_df, ["Cancer"], required=False)
        r_db_col = get_col(random_df, ["database", "Database"], required=False)

        if s_cancer_col is None or s_db_col is None or r_cancer_col is None or r_db_col is None:
            raise ValueError("Cannot compute SF4 summary without Cancer and database columns.")

        ssum = structured.groupby([s_cancer_col, s_db_col])[d_col_s].agg(
            Structured_mean_D="mean",
            Structured_median_D="median",
            Structured_max_D="max"
        ).reset_index().rename(columns={s_cancer_col: "Cancer", s_db_col: "Database"})

        rsum = random_df.groupby([r_cancer_col, r_db_col])[d_col_r].agg(
            Random_mean_D="mean",
            Random_median_D="median",
            Random_95pct_D=lambda x: np.nanpercentile(x, 95),
            Random_max_D="max"
        ).reset_index().rename(columns={r_cancer_col: "Cancer", r_db_col: "Database"})

        summary = ssum.merge(rsum, on=["Cancer", "Database"], how="inner")
        summary["Structured_above_random95_fraction"] = np.nan

    # normalize summary column names
    if "database" in summary.columns and "Database" not in summary.columns:
        summary = summary.rename(columns={"database": "Database"})

    summary = clean_numeric(summary, [
        "Structured_mean_D", "Structured_median_D", "Structured_max_D",
        "Random_mean_D", "Random_median_D", "Random_95pct_D", "Random_max_D",
        "Structured_above_random95_fraction"
    ])

    # B. mean structured vs random, aggregate by cancer
    ax = fig.add_subplot(gs[0, 1])
    if {"Cancer", "Structured_mean_D", "Random_mean_D"}.issubset(summary.columns):
        by_cancer = summary.groupby("Cancer", as_index=False)[["Structured_mean_D", "Random_mean_D"]].mean()
        cancers = order_cancers(by_cancer["Cancer"])
        by_cancer = by_cancer.set_index("Cancer").reindex(cancers).reset_index()
        x = np.arange(len(by_cancer))
        width = 0.38
        ax.bar(x - width/2, by_cancer["Structured_mean_D"], width, label="Structured mean D")
        ax.bar(x + width/2, by_cancer["Random_mean_D"], width, label="Random mean D")
        ax.set_xticks(x)
        ax.set_xticklabels(by_cancer["Cancer"], rotation=60, fontsize=8)
        ax.set_ylabel("Mean D")
        ax.set_title("Cancer-wise mean discriminability")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "Summary columns unavailable", ha="center", va="center")
        ax.set_axis_off()
    panel(ax, "B")

    # C. fraction above random 95
    ax = fig.add_subplot(gs[1, 0])
    frac_col = get_col(summary, ["Structured_above_random95_fraction", "Fraction_above_random95"], required=False)
    if frac_col is not None and "Cancer" in summary.columns:
        frac = summary.groupby("Cancer", as_index=False)[frac_col].mean()
        cancers = order_cancers(frac["Cancer"])
        frac = frac.set_index("Cancer").reindex(cancers).reset_index()
        ax.bar(np.arange(len(frac)), frac[frac_col])
        ax.set_xticks(np.arange(len(frac)))
        ax.set_xticklabels(frac["Cancer"], rotation=60, fontsize=8)
        ax.set_ylabel("Fraction")
        ax.set_title("Structured pathways above random 95th percentile")
    else:
        ax.text(0.5, 0.5, "Fraction column unavailable", ha="center", va="center")
        ax.set_axis_off()
    panel(ax, "C")

    # D. random 95 vs structured max
    ax = fig.add_subplot(gs[1, 1])
    if {"Random_95pct_D", "Structured_max_D"}.issubset(summary.columns):
        ax.scatter(summary["Random_95pct_D"], summary["Structured_max_D"], s=70, alpha=0.75)
        lim_max = np.nanmax([summary["Random_95pct_D"].max(), summary["Structured_max_D"].max()])
        if np.isfinite(lim_max):
            ax.plot([0, lim_max], [0, lim_max], linestyle="--", linewidth=1)
        ax.set_xlabel("Random 95th percentile D")
        ax.set_ylabel("Structured max D")
        ax.set_title("Structured maxima relative to random baseline")
    else:
        ax.text(0.5, 0.5, "Required columns unavailable", ha="center", va="center")
        ax.set_axis_off()
    panel(ax, "D")

    fig.suptitle(
        "Supplementary Figure 4. Structured-versus-random discriminability distributions",
        fontsize=16,
        fontweight="bold"
    )

    savefig("SF4_structured_vs_random_distributions")


# ============================================================
# 3. Supplementary Figure 5
# Pathway size versus uncertainty width
# ============================================================

def make_sf5():
    """
    SF5:
      A. All pathway size vs W
      B. Median W by mapped-size bins
      C. Size vs coherence
      D. Database-level pathway-size distribution

    Input:
      SF5_pathway_size_vs_W.csv
    """

    df = read_required("SF5_pathway_size_vs_W.csv")

    required = ["Cancer", "database", "n_genes_mapped", "W_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SF5 missing columns: {missing}")

    df = clean_numeric(df, ["n_genes_mapped", "W_mean", "W_median", "W_sd", "mu_mean_abs", "coherence_mean"])
    df = df.dropna(subset=["database", "n_genes_mapped", "W_mean"])
    df = df[(df["n_genes_mapped"] > 0) & (df["W_mean"] > 0)]

    dbs = order_databases(df["database"])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)

    # A. scatter size vs W
    ax = fig.add_subplot(gs[0, 0])
    for db in dbs:
        sub = df[df["database"] == db]
        if not sub.empty:
            ax.scatter(sub["n_genes_mapped"], sub["W_mean"], s=18, alpha=0.45, label=db)
    ax.set_xscale("log")
    ax.set_xlabel("Mapped pathway size")
    ax.set_ylabel("Mean W (FWHM)")
    ax.set_title("Pathway size versus uncertainty width")
    ax.legend(frameon=False, fontsize=8)
    panel(ax, "A")

    # B. binned median W
    ax = fig.add_subplot(gs[0, 1])
    bins = [0, 10, 20, 50, 100, 200, 300, 500, 1000, np.inf]
    labels = ["≤10", "11–20", "21–50", "51–100", "101–200", "201–300", "301–500", "501–1000", ">1000"]
    df["size_bin"] = pd.cut(df["n_genes_mapped"], bins=bins, labels=labels, include_lowest=True)
    binned = df.groupby("size_bin", observed=False)["W_mean"].median().reset_index()
    ax.bar(np.arange(len(binned)), binned["W_mean"])
    ax.set_xticks(np.arange(len(binned)))
    ax.set_xticklabels(binned["size_bin"].astype(str), rotation=45, fontsize=8)
    ax.set_ylabel("Median W")
    ax.set_title("Median uncertainty width by pathway-size bin")
    panel(ax, "B")

    # C. size vs coherence
    ax = fig.add_subplot(gs[1, 0])
    if "coherence_mean" in df.columns:
        subdf = df.dropna(subset=["coherence_mean"])
        for db in dbs:
            sub = subdf[subdf["database"] == db]
            if not sub.empty:
                ax.scatter(sub["n_genes_mapped"], sub["coherence_mean"], s=18, alpha=0.45, label=db)
        ax.set_xscale("log")
        ax.set_xlabel("Mapped pathway size")
        ax.set_ylabel("Mean coherence")
        ax.set_title("Pathway size versus coherence")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "coherence_mean unavailable", ha="center", va="center")
        ax.set_axis_off()
    panel(ax, "C")

    # D. size distribution by database
    ax = fig.add_subplot(gs[1, 1])
    data = [df[df["database"] == db]["n_genes_mapped"].dropna().values for db in dbs]
    data = [x for x in data if len(x) > 0]
    labels_present = [db for db in dbs if len(df[df["database"] == db]) > 0]
    ax.boxplot(data, labels=labels_present, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Mapped pathway size")
    ax.set_title("Mapped pathway-size distribution by database")
    ax.tick_params(axis="x", rotation=30)
    panel(ax, "D")

    fig.suptitle(
        "Supplementary Figure 5. Pathway-size effects on uncertainty-width estimation",
        fontsize=16,
        fontweight="bold"
    )

    savefig("SF5_pathway_size_vs_uncertainty_width")


# ============================================================
# 4. Supplementary Figure 6
# K–λ heatmaps and optimal-scale summary
# ============================================================

def make_sf6():
    """
    SF6:
      A. Cancer x K heatmap of D(K)
      B. Best K per cancer
      C. Best λ per cancer
      D. Max D per cancer

    Inputs:
      SF3_K_scale_all_cancers.csv
      SF6_K_lambda_heatmap_table.csv [optional, can be recomputed]
      SF10_best_K_lambda_summary.csv [optional, can be recomputed]
    """

    kscale = read_optional("SF3_K_scale_all_cancers.csv")
    if kscale is None:
        raise FileNotFoundError("SF6 requires SF3_K_scale_all_cancers.csv")

    required = ["Cancer", "K", "lambda", "D_clinical"]
    missing = [c for c in required if c not in kscale.columns]
    if missing:
        raise ValueError(f"SF6 missing columns in kscale: {missing}")

    kscale = clean_numeric(kscale, ["K", "lambda", "D_clinical"])
    kscale = kscale.dropna(subset=["Cancer", "K", "lambda", "D_clinical"])

    cancers = order_cancers(kscale["Cancer"])
    k_values = sorted(kscale["K"].dropna().unique())

    # D(K) = max over lambda
    dk = (
        kscale.sort_values("D_clinical", ascending=False)
        .groupby(["Cancer", "K"], as_index=False)
        .first()
        .sort_values(["Cancer", "K"])
    )

    matrix = dk.pivot_table(index="Cancer", columns="K", values="D_clinical", aggfunc="max")
    matrix = matrix.reindex(cancers)

    # best K/lambda per cancer
    best = (
        kscale.sort_values("D_clinical", ascending=False)
        .groupby("Cancer", as_index=False)
        .first()
        .set_index("Cancer")
        .reindex(cancers)
        .reset_index()
    )

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)

    # A. heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(matrix[k_values].values, aspect="auto")
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels([str(int(k)) for k in k_values], rotation=45)
    ax.set_yticks(np.arange(len(cancers)))
    ax.set_yticklabels(cancers, fontsize=8)
    ax.set_xlabel("Observation scale K")
    ax.set_ylabel("Cancer")
    ax.set_title("D(K) heatmap across cancers")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Best D across λ")

    # mark K*
    for i, cancer in enumerate(cancers):
        row = best[best["Cancer"] == cancer]
        if row.empty or pd.isna(row["K"].iloc[0]):
            continue
        kstar = row["K"].iloc[0]
        if kstar in k_values:
            j = k_values.index(kstar)
            ax.scatter(j, i, marker="s", s=60, facecolors="none", edgecolors="white", linewidths=1.5)
    panel(ax, "A")

    # B. best K
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(best))
    ax.bar(x, best["K"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(best["Cancer"], rotation=60, fontsize=8)
    ax.set_ylabel("Optimal K*")
    ax.set_title("Optimal observation scale")
    panel(ax, "B")

    # C. best lambda
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(x, best["lambda"])
    ax.set_xticks(x)
    ax.set_xticklabels(best["Cancer"], rotation=60, fontsize=8)
    ax.set_ylabel("Optimal λ*")
    ax.set_title("Optimal uncertainty penalty")
    panel(ax, "C")

    # D. max D
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(x, best["D_clinical"])
    ax.set_xticks(x)
    ax.set_xticklabels(best["Cancer"], rotation=60, fontsize=8)
    ax.set_ylabel("Max D")
    ax.set_title("Maximum scale-dependent discriminability")
    panel(ax, "D")

    fig.suptitle(
        "Supplementary Figure 6. K–λ interaction and optimal observation-scale summary",
        fontsize=16,
        fontweight="bold"
    )

    savefig("SF6_K_lambda_heatmap_and_optimal_summary")


# ============================================================
# 5. Main
# ============================================================

def main():
    log("============================================================")
    log("AIDO-D-PHY Supplementary Figures 4–6 Builder START")
    log(f"INPUT_DIR = {INPUT_DIR}")
    log(f"OUT_DIR   = {OUT_DIR}")
    log("============================================================")

    make_sf4()
    make_sf5()
    make_sf6()

    log("============================================================")
    log("DONE.")
    log(f"Check output folder: {OUT_DIR}")
    log("============================================================")


if __name__ == "__main__":
    main()
