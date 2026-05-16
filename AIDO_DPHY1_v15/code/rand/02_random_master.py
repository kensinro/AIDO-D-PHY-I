# ============================================================
# AIDO-D RANDOM TEST MASTER PIPELINE FORMAL v1.3
# Layer 1: Structured vs Random
# Layer 2: High-D Random Dissection
# Layer 3: Cross-Cancer Recurrence
# Formal run: 18 cancers, 5 databases
# ============================================================

import os
import math
import time
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

try:
    from lifelines.statistics import logrank_test
except ImportError:
    raise ImportError("Please install lifelines first: pip install lifelines")

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIR = Path(r"D:/AIDO-Data/UCSC_XENA")
GSEA_DIR = Path(r"D:/AIDO-Data/GSEA")
OUT_DIR = Path(r"D:/AIDO-Temp/AIDO-D-RANDOM-TEST-FORMAL")

CANCERS = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC",
    "LAML", "LIHC", "LUAD", "LUSC", "OV", "PAAD",
    "PRAD", "SARC", "SKCM", "STAD", "THCA", "UCEC"
]

DATABASES_TO_RUN = [
    "Hallmark",
    "GO_BP",
    "Reactome",
    "KEGG_Medicus",
    "WikiPathways"
]

N_RANDOM_PER_STRUCTURED_SET = 20

CANCER_FOLDER_MAP = {
    "BLCA": "Bladder Cancer (BLCA)",
    "BRCA": "Breast Cancer (BRCA)",
    "COAD": "Colon Cancer (COAD)",
    "GBM": "Glioblastoma (GBM)",
    "HNSC": "Head and Neck Cancer (HNSC)",
    "KIRC": "Kidney Clear Cell Carcinoma (KIRC)",
    "LAML": "Acute Myeloid Leukemia (LAML)",
    "LIHC": "Liver Cancer (LIHC)",
    "LUAD": "Lung Adenocarcinoma (LUAD)",
    "LUSC": "Lung Squamous Cell Carcinoma (LUSC)",
    "OV": "Ovarian Cancer (OV)",
    "PAAD": "Pancreatic Cancer (PAAD)",
    "PRAD": "Prostate Cancer (PRAD)",
    "SARC": "Sarcoma (SARC)",
    "SKCM": "Melanoma (SKCM)",
    "STAD": "Stomach Cancer (STAD)",
    "THCA": "Thyroid Cancer (THCA)",
    "UCEC": "Endometrioid Cancer (UCEC)",
}

GMT_FILES = {
    "Hallmark": GSEA_DIR / "h.all.v2026.1.Hs.symbols.gmt",
    "GO_BP": GSEA_DIR / "c5.go.bp.v2026.1.Hs.symbols.gmt",
    "Reactome": GSEA_DIR / "c2.cp.reactome.v2026.1.Hs.symbols.gmt",
    "KEGG_Medicus": GSEA_DIR / "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt",
    "WikiPathways": GSEA_DIR / "c2.cp.wikipathways.v2026.1.Hs.symbols.gmt",
}

RANDOM_SEED = 20260508

MIN_MAPPED_GENES = 10
MAX_MAPPED_GENES = 500

HIGH_D_PERCENTILE = 95
TOP_N_RANDOM_SAVE = 100

GLOBAL_START_TIME = time.time()


# ============================================================
# UTILITIES
# ============================================================

def log(msg):
    elapsed = (time.time() - GLOBAL_START_TIME) / 60
    print(f"[{elapsed:8.2f} min] {msg}", flush=True)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def progress_message(cancer, db_name, current, total, mode="Progress", extra=""):
    pct = 100 * current / max(total, 1)
    log(f"[{mode}] {cancer} | {db_name} | {current}/{total} ({pct:.2f}%) {extra}")


def clean_sample_id(x):
    return str(x).replace(".", "-")[:12]


def clean_gene_symbol(x):
    x = str(x).strip()
    if "|" in x:
        x = x.split("|")[0]
    return x.upper()


def safe_neglog10(p):
    if p is None or pd.isna(p):
        return np.nan
    return -math.log10(max(float(p), 1e-300))


def read_table_robust(file_path, sep="\t"):
    encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding=enc)
            log(f"[INFO] Loaded {file_path.name} with encoding: {enc}")
            return df
        except Exception as e:
            last_error = e

    try:
        df = pd.read_csv(file_path, sep=sep, engine="python")
        log(f"[WARNING] Loaded {file_path.name} using python engine fallback")
        return df
    except Exception:
        raise last_error


def find_existing_file(folder, candidates):
    for c in candidates:
        p = folder / c
        if p.exists():
            return p
    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_gmt(gmt_path):
    gene_sets = {}

    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            name = parts[0]
            genes = [clean_gene_symbol(g) for g in parts[2:] if g.strip()]
            gene_sets[name] = sorted(set(genes))

    return gene_sets


def load_expression_matrix(cancer_dir):
    ge_file = find_existing_file(cancer_dir, ["GE.tsv", "HiSeqV2.tsv", "expression.tsv"])

    if ge_file is None:
        raise FileNotFoundError(f"No GE.tsv found in {cancer_dir}")

    log(f"Loading GE: {ge_file}")

    df = read_table_robust(ge_file, sep="\t")
    df = df.set_index(df.columns[0])

    df.index = [clean_gene_symbol(g) for g in df.index]
    df = df.groupby(df.index).mean()

    df.columns = [clean_sample_id(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    return df


def load_survival(cancer_dir):
    survival_file = None

    for f in cancer_dir.glob("*survival*.tsv"):
        survival_file = f
        break

    if survival_file is None:
        survival_file = find_existing_file(
            cancer_dir,
            ["Phenotype.tsv", "phenotype.tsv", "clinical.tsv"]
        )

    if survival_file is None:
        raise FileNotFoundError(f"No survival/phenotype file found in {cancer_dir}")

    log(f"Loading survival: {survival_file}")

    sv = read_table_robust(survival_file, sep="\t")

    sample_col = None
    for c in sv.columns:
        cl = str(c).lower()
        if "sample" in cl or "patient" in cl or "barcode" in cl or c in ["_PATIENT", "sampleID", "SampleID"]:
            sample_col = c
            break

    if sample_col is None:
        sample_col = sv.columns[0]

    sv["sample_id"] = sv[sample_col].apply(clean_sample_id)

    time_candidates = [
        "OS.time", "OS_Time", "OS_Time_nature2012",
        "OS.time.days", "OS_days",
        "time", "days_to_death",
        "DSS.time", "PFI.time"
    ]

    event_candidates = [
        "OS", "OS_event", "OS_event_nature2012",
        "event", "status", "death",
        "DSS", "PFI"
    ]

    time_col = None
    event_col = None

    for c in time_candidates:
        if c in sv.columns:
            time_col = c
            break

    for c in event_candidates:
        if c in sv.columns:
            event_col = c
            break

    if time_col is None:
        possible = [c for c in sv.columns if "time" in str(c).lower() or "days" in str(c).lower()]
        if possible:
            time_col = possible[0]

    if event_col is None:
        possible = [
            c for c in sv.columns
            if "event" in str(c).lower()
            or "status" in str(c).lower()
            or "death" in str(c).lower()
        ]
        if possible:
            event_col = possible[0]

    if time_col is None or event_col is None:
        raise ValueError(
            f"Cannot identify survival columns in {survival_file}\n"
            f"Columns: {list(sv.columns)}"
        )

    out = sv[["sample_id", time_col, event_col]].copy()
    out.columns = ["sample_id", "time", "event"]

    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["event"] = pd.to_numeric(out["event"], errors="coerce")

    out = out.dropna(subset=["sample_id", "time", "event"])
    out = out[out["time"] > 0]

    unique_events = sorted(out["event"].dropna().unique())
    if set(unique_events).issubset({1, 2}):
        out["event"] = out["event"].replace({1: 0, 2: 1})

    out["event"] = out["event"].astype(int)
    out = out.drop_duplicates("sample_id")

    log(f"[INFO] Survival columns used: time={time_col}, event={event_col}")

    return out


def zscore_expression(expr):
    log("Z-scoring expression matrix...")
    mean = expr.mean(axis=1)
    std = expr.std(axis=1).replace(0, np.nan)
    z = expr.sub(mean, axis=0).div(std, axis=0)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0)


# ============================================================
# SCORE + SURVIVAL
# ============================================================

def compute_gene_set_score(zexpr, genes):
    mapped = [g for g in genes if g in zexpr.index]

    if len(mapped) < MIN_MAPPED_GENES:
        return None, mapped

    score = zexpr.loc[mapped].mean(axis=0)
    return score, mapped


def survival_D(score_series, survival_df):
    df = pd.DataFrame({
        "sample_id": score_series.index,
        "score": score_series.values
    })

    merged = df.merge(survival_df, on="sample_id", how="inner")
    merged = merged.dropna(subset=["score", "time", "event"])

    if merged.shape[0] < 30:
        return np.nan, np.nan, merged.shape[0], np.nan

    if merged["event"].sum() < 5:
        return np.nan, np.nan, merged.shape[0], merged["event"].sum()

    med = merged["score"].median()

    low = merged[merged["score"] <= med]
    high = merged[merged["score"] > med]

    if low.shape[0] < 10 or high.shape[0] < 10:
        return np.nan, np.nan, merged.shape[0], merged["event"].sum()

    try:
        res = logrank_test(
            low["time"],
            high["time"],
            event_observed_A=low["event"],
            event_observed_B=high["event"]
        )

        p = float(res.p_value)
        D = safe_neglog10(p)

        return p, D, merged.shape[0], merged["event"].sum()

    except Exception:
        return np.nan, np.nan, merged.shape[0], merged["event"].sum()


# ============================================================
# LAYER 1
# ============================================================

def run_structured_for_database(cancer, db_name, gene_sets, zexpr, survival_df, out_db_dir):
    log(f"Starting structured analysis: {cancer} | {db_name}")

    rows = []
    total_sets = len(gene_sets)

    for i, (gs_name, genes) in enumerate(gene_sets.items(), start=1):
        if i == 1 or i % 100 == 0 or i == total_sets:
            progress_message(cancer, db_name, i, total_sets, mode="Structured")

        score, mapped = compute_gene_set_score(zexpr, genes)

        if score is None:
            continue

        if len(mapped) > MAX_MAPPED_GENES:
            continue

        p, D, n, events = survival_D(score, survival_df)

        rows.append({
            "Cancer": cancer,
            "Database": db_name,
            "Type": "Structured",
            "GeneSet": gs_name,
            "OriginalSize": len(genes),
            "MappedSize": len(mapped),
            "N": n,
            "Events": events,
            "p_logrank": p,
            "D": D,
            "Genes": ";".join(mapped)
        })

    df = pd.DataFrame(rows)
    out_file = out_db_dir / f"{cancer}_{db_name}_structured_D.csv"
    df.to_csv(out_file, index=False)

    log(f"Saved structured result: {out_file}")

    return df


def run_random_matched_for_database(cancer, db_name, structured_df, zexpr, survival_df, out_db_dir):
    log(f"Starting random matched analysis: {cancer} | {db_name}")

    rng = random.Random(RANDOM_SEED + abs(hash(cancer + db_name)) % 1000000)
    universe = sorted(zexpr.index)

    rows = []

    high_gene_dir = out_db_dir / "highD_random_gene_lists"
    ensure_dir(high_gene_dir)

    if structured_df.empty:
        rdf = pd.DataFrame()
        rdf.to_csv(out_db_dir / f"{cancer}_{db_name}_random_D.csv", index=False)
        return rdf

    valid_struct = structured_df.dropna(subset=["D"])
    valid_struct = valid_struct[valid_struct["MappedSize"] >= MIN_MAPPED_GENES]

    total_struct = len(valid_struct)
    total_random_tests = total_struct * N_RANDOM_PER_STRUCTURED_SET
    finished_random_tests = 0

    log(
        f"Random workload: {cancer} | {db_name} | "
        f"{total_struct} structured sets x {N_RANDOM_PER_STRUCTURED_SET} "
        f"= {total_random_tests} random tests"
    )

    for idx, (_, row) in enumerate(valid_struct.iterrows(), start=1):
        if idx == 1 or idx % 50 == 0 or idx == total_struct:
            progress_message(
                cancer,
                db_name,
                idx,
                total_struct,
                mode="Random-Set",
                extra=f"| finished={finished_random_tests}/{total_random_tests}"
            )

        size = int(row["MappedSize"])

        if size < MIN_MAPPED_GENES or size > len(universe):
            continue

        for r in range(N_RANDOM_PER_STRUCTURED_SET):
            finished_random_tests += 1

            if finished_random_tests == 1 or finished_random_tests % 500 == 0 or finished_random_tests == total_random_tests:
                progress_message(
                    cancer,
                    db_name,
                    finished_random_tests,
                    total_random_tests,
                    mode="Random-Test",
                    extra=f"| matched={row['GeneSet']} | iter={r+1}/{N_RANDOM_PER_STRUCTURED_SET}"
                )

            genes = rng.sample(universe, size)
            score, mapped = compute_gene_set_score(zexpr, genes)

            if score is None:
                continue

            p, D, n, events = survival_D(score, survival_df)

            random_id = f"{row['GeneSet']}__RAND{r+1:03d}"

            rows.append({
                "Cancer": cancer,
                "Database": db_name,
                "Type": "Random",
                "MatchedStructuredGeneSet": row["GeneSet"],
                "GeneSet": random_id,
                "OriginalSize": size,
                "MappedSize": len(mapped),
                "N": n,
                "Events": events,
                "p_logrank": p,
                "D": D,
                "Genes": ";".join(mapped)
            })

    rdf = pd.DataFrame(rows)
    out_file = out_db_dir / f"{cancer}_{db_name}_random_D.csv"
    rdf.to_csv(out_file, index=False)

    log(f"Saved random result: {out_file}")

    if rdf.empty or not rdf["D"].notna().any():
        return rdf

    threshold = np.nanpercentile(rdf["D"].dropna(), HIGH_D_PERCENTILE)

    high_df = rdf[rdf["D"] >= threshold].copy()
    high_df = high_df.sort_values("D", ascending=False).head(TOP_N_RANDOM_SAVE)

    high_file = out_db_dir / f"{cancer}_{db_name}_top_highD_random_sets.csv"
    high_df.to_csv(high_file, index=False)

    log(f"Saved high-D random sets: {high_file}")

    for i, (_, r) in enumerate(high_df.iterrows(), start=1):
        fname = high_gene_dir / f"{cancer}_{db_name}_highD_random_{i:03d}_D{r['D']:.3f}.txt"
        with open(fname, "w") as f:
            f.write(str(r["Genes"]).replace(";", "\n"))

    return rdf


def summarize_structured_vs_random(cancer, db_name, sdf, rdf, out_db_dir):
    log(f"Summarizing Layer 1: {cancer} | {db_name}")

    summary = {
        "Cancer": cancer,
        "Database": db_name,
        "N_structured": len(sdf),
        "N_random": len(rdf),
        "Structured_mean_D": sdf["D"].mean() if not sdf.empty else np.nan,
        "Structured_median_D": sdf["D"].median() if not sdf.empty else np.nan,
        "Structured_max_D": sdf["D"].max() if not sdf.empty else np.nan,
        "Random_mean_D": rdf["D"].mean() if not rdf.empty else np.nan,
        "Random_median_D": rdf["D"].median() if not rdf.empty else np.nan,
        "Random_95pct_D": np.nanpercentile(rdf["D"].dropna(), 95)
        if not rdf.empty and rdf["D"].notna().any() else np.nan,
        "Random_max_D": rdf["D"].max() if not rdf.empty else np.nan,
        "Structured_above_random95_count": np.nan,
        "Structured_above_random95_fraction": np.nan
    }

    if not sdf.empty and not rdf.empty and rdf["D"].notna().any():
        r95 = summary["Random_95pct_D"]
        summary["Structured_above_random95_count"] = int((sdf["D"] >= r95).sum())
        summary["Structured_above_random95_fraction"] = float((sdf["D"] >= r95).mean())

    out = pd.DataFrame([summary])
    out_file = out_db_dir / f"{cancer}_{db_name}_structured_vs_random_summary.csv"
    out.to_csv(out_file, index=False)

    return out


# ============================================================
# LAYER 2
# ============================================================

def overlap_with_structured_sets(random_df, structured_sets_all, out_file, cancer, db_name):
    log(f"Starting Layer 2 high-D random dissection: {cancer} | {db_name}")

    rows = []

    if random_df.empty or not random_df["D"].notna().any():
        pd.DataFrame().to_csv(out_file, index=False)
        return pd.DataFrame()

    threshold = np.nanpercentile(random_df["D"].dropna(), HIGH_D_PERCENTILE)
    high_df = random_df[random_df["D"] >= threshold].sort_values("D", ascending=False)

    universe_genes = set()

    for _, gs_dict in structured_sets_all.items():
        for genes in gs_dict.values():
            universe_genes.update(genes)

    M = len(universe_genes)
    total_high = len(high_df)

    for i, (_, r) in enumerate(high_df.iterrows(), start=1):
        if i == 1 or i % 20 == 0 or i == total_high:
            progress_message(cancer, db_name, i, total_high, mode="Dissection")

        rgenes = set(str(r["Genes"]).split(";"))
        n = len(rgenes)

        best_hits = []

        for db, gs_dict in structured_sets_all.items():
            for gs_name, genes in gs_dict.items():
                sgenes = set(genes)
                k = len(rgenes & sgenes)

                if k == 0:
                    continue

                K = len(sgenes)
                p_overlap = hypergeom.sf(k - 1, M, K, n)
                jaccard = k / len(rgenes | sgenes)

                best_hits.append({
                    "Cancer": r["Cancer"],
                    "RandomGeneSet": r["GeneSet"],
                    "RandomD": r["D"],
                    "RandomSize": n,
                    "MatchedStructuredGeneSet": r.get("MatchedStructuredGeneSet", ""),
                    "OverlapDB": db,
                    "OverlapGeneSet": gs_name,
                    "OverlapCount": k,
                    "Jaccard": jaccard,
                    "OverlapP": p_overlap,
                    "OverlapGenes": ";".join(sorted(rgenes & sgenes))
                })

        best_hits = sorted(best_hits, key=lambda x: (x["OverlapP"], -x["OverlapCount"]))[:10]
        rows.extend(best_hits)

    out = pd.DataFrame(rows)
    out.to_csv(out_file, index=False)

    log(f"Saved Layer 2 dissection: {out_file}")

    return out


def summarize_highD_random_dissection(cancer, db_name, random_df, overlap_df, out_db_dir):
    if random_df.empty or not random_df["D"].notna().any():
        return pd.DataFrame()

    threshold = np.nanpercentile(random_df["D"].dropna(), HIGH_D_PERCENTILE)
    high_df = random_df[random_df["D"] >= threshold].copy()

    summary = {
        "Cancer": cancer,
        "Database": db_name,
        "HighD_threshold": threshold,
        "N_highD_random": len(high_df),
        "Mean_highD_random_D": high_df["D"].mean(),
        "Max_highD_random_D": high_df["D"].max(),
        "N_with_structured_overlap": 0,
        "Best_overlap_p": np.nan,
        "Best_overlap_gene_set": ""
    }

    if overlap_df is not None and not overlap_df.empty:
        summary["N_with_structured_overlap"] = overlap_df["RandomGeneSet"].nunique()
        best = overlap_df.sort_values("OverlapP").iloc[0]
        summary["Best_overlap_p"] = best["OverlapP"]
        summary["Best_overlap_gene_set"] = f"{best['OverlapDB']}::{best['OverlapGeneSet']}"

    out = pd.DataFrame([summary])
    out_file = out_db_dir / f"{cancer}_{db_name}_highD_random_dissection_summary.csv"
    out.to_csv(out_file, index=False)

    return out


# ============================================================
# LAYER 3
# ============================================================

def build_cross_cancer_recurrence(all_high_random_files, out_dir):
    log("Starting Layer 3 cross-cancer recurrence analysis")

    rows = []

    for i, file in enumerate(all_high_random_files, start=1):
        log(f"Reading high-D random file {i}/{len(all_high_random_files)}: {file.name}")

        df = pd.read_csv(file)

        if df.empty:
            continue

        for _, r in df.iterrows():
            genes = sorted(str(r["Genes"]).split(";"))
            gene_signature = "|".join(genes)

            rows.append({
                "Cancer": r["Cancer"],
                "Database": r["Database"],
                "GeneSet": r["GeneSet"],
                "D": r["D"],
                "p_logrank": r["p_logrank"],
                "MappedSize": r["MappedSize"],
                "GeneSignature": gene_signature,
                "Genes": ";".join(genes)
            })

    all_df = pd.DataFrame(rows)

    if all_df.empty:
        all_df.to_csv(out_dir / "cross_cancer_highD_random_all.csv", index=False)
        return all_df, pd.DataFrame()

    all_df.to_csv(out_dir / "cross_cancer_highD_random_all.csv", index=False)

    exact = (
        all_df.groupby("GeneSignature")
        .agg(
            RecurrenceCount=("Cancer", "nunique"),
            Cancers=("Cancer", lambda x: ";".join(sorted(set(x)))),
            MeanD=("D", "mean"),
            MaxD=("D", "max"),
            Size=("MappedSize", "first"),
            Genes=("Genes", "first")
        )
        .reset_index()
        .sort_values(["RecurrenceCount", "MaxD"], ascending=False)
    )

    exact.to_csv(out_dir / "cross_cancer_exact_random_signature_recurrence.csv", index=False)

    gene_rows = []

    for cancer, sub in all_df.groupby("Cancer"):
        counter = Counter()

        for genes in sub["Genes"]:
            counter.update(str(genes).split(";"))

        for gene, count in counter.items():
            gene_rows.append({
                "Cancer": cancer,
                "Gene": gene,
                "HighDSetCount": count
            })

    gene_df = pd.DataFrame(gene_rows)

    if not gene_df.empty:
        gene_summary = (
            gene_df.groupby("Gene")
            .agg(
                CancerCount=("Cancer", "nunique"),
                TotalHighDSetCount=("HighDSetCount", "sum"),
                Cancers=("Cancer", lambda x: ";".join(sorted(set(x))))
            )
            .reset_index()
            .sort_values(["CancerCount", "TotalHighDSetCount"], ascending=False)
        )

        gene_summary.to_csv(
            out_dir / "cross_cancer_gene_recurrence_in_highD_random.csv",
            index=False
        )
    else:
        gene_summary = pd.DataFrame()

    log("Finished Layer 3 cross-cancer recurrence")

    return exact, gene_summary


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_dir(OUT_DIR)

    log("============================================================")
    log("AIDO-D RANDOM TEST FORMAL PIPELINE v1.3 START")
    log(f"CANCERS = {CANCERS}")
    log(f"DATABASES_TO_RUN = {DATABASES_TO_RUN}")
    log(f"N_RANDOM_PER_STRUCTURED_SET = {N_RANDOM_PER_STRUCTURED_SET}")
    log(f"OUT_DIR = {OUT_DIR}")
    log("============================================================")

    all_gene_sets = {}

    log("Loading GMT databases...")

    for db_name, gmt_path in GMT_FILES.items():
        if db_name not in DATABASES_TO_RUN:
            continue

        if not gmt_path.exists():
            log(f"[WARNING] Missing GMT: {gmt_path}")
            continue

        all_gene_sets[db_name] = load_gmt(gmt_path)
        log(f"{db_name}: {len(all_gene_sets[db_name])} gene sets loaded")

    all_summary = []
    all_dissection_summary = []
    all_high_random_files = []

    for cancer_index, cancer in enumerate(CANCERS, start=1):
        log("")
        log("=" * 80)
        log(f"Running cancer {cancer_index}/{len(CANCERS)}: {cancer}")
        log("=" * 80)

        cancer_dir = BASE_DIR / CANCER_FOLDER_MAP[cancer]

        if not cancer_dir.exists():
            log(f"[SKIP] Cancer directory not found: {cancer_dir}")
            continue

        cancer_out = OUT_DIR / f"AIDO-D-RandomTest-{cancer}"
        ensure_dir(cancer_out)

        try:
            expr = load_expression_matrix(cancer_dir)
            survival = load_survival(cancer_dir)
        except Exception as e:
            log(f"[SKIP] Failed loading {cancer}: {e}")
            continue

        common_samples = sorted(set(expr.columns) & set(survival["sample_id"]))

        expr = expr[common_samples]
        survival = survival[survival["sample_id"].isin(common_samples)].copy()

        log(f"{cancer}: expression genes={expr.shape[0]}, samples={expr.shape[1]}")
        log(f"{cancer}: survival samples={survival.shape[0]}, events={survival['event'].sum()}")

        if expr.shape[1] < 30 or survival["event"].sum() < 5:
            log(f"[WARNING] Very low sample/event number for {cancer}")

        zexpr = zscore_expression(expr)

        for db_index, (db_name, gene_sets) in enumerate(all_gene_sets.items(), start=1):
            log("")
            log(f"--- {cancer} | {db_name} ({db_index}/{len(all_gene_sets)}) ---")

            db_out = cancer_out / db_name
            ensure_dir(db_out)

            sdf = run_structured_for_database(
                cancer, db_name, gene_sets, zexpr, survival, db_out
            )

            rdf = run_random_matched_for_database(
                cancer, db_name, sdf, zexpr, survival, db_out
            )

            summary = summarize_structured_vs_random(
                cancer, db_name, sdf, rdf, db_out
            )

            all_summary.append(summary)

            high_file = db_out / f"{cancer}_{db_name}_top_highD_random_sets.csv"

            if high_file.exists():
                all_high_random_files.append(high_file)

            overlap_file = db_out / f"{cancer}_{db_name}_highD_random_overlap_dissection.csv"

            overlap_df = overlap_with_structured_sets(
                rdf, all_gene_sets, overlap_file, cancer, db_name
            )

            dis_summary = summarize_highD_random_dissection(
                cancer, db_name, rdf, overlap_df, db_out
            )

            if dis_summary is not None and not dis_summary.empty:
                all_dissection_summary.append(dis_summary)

        log(f"Finished cancer: {cancer}")

    if all_summary:
        global_summary = pd.concat(all_summary, ignore_index=True)
        out_file = OUT_DIR / "GLOBAL_Layer1_structured_vs_random_summary.csv"
        global_summary.to_csv(out_file, index=False)
        log(f"Saved GLOBAL Layer 1 summary: {out_file}")

    if all_dissection_summary:
        global_dissection = pd.concat(all_dissection_summary, ignore_index=True)
        out_file = OUT_DIR / "GLOBAL_Layer2_highD_random_dissection_summary.csv"
        global_dissection.to_csv(out_file, index=False)
        log(f"Saved GLOBAL Layer 2 summary: {out_file}")

    recurrence_out = OUT_DIR / "GLOBAL_Layer3_cross_cancer_recurrence"
    ensure_dir(recurrence_out)

    build_cross_cancer_recurrence(all_high_random_files, recurrence_out)

    log("")
    log("============================================================")
    log("FORMAL RUN DONE")
    log(f"Output folder: {OUT_DIR}")
    log("Main outputs:")
    log("1. GLOBAL_Layer1_structured_vs_random_summary.csv")
    log("2. GLOBAL_Layer2_highD_random_dissection_summary.csv")
    log("3. GLOBAL_Layer3_cross_cancer_recurrence/")
    log("============================================================")


if __name__ == "__main__":
    main()