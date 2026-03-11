"""
SALI (Structure-Activity Landscape Index) Calculator

Identifies activity cliffs in drug discovery datasets.

Formula: SALI(i,j) = |pIC50_i - pIC50_j| / (1 - Sim(i,j) + ε)
ε = 0.001

Reference: Guha & Van Drie, J. Chem. Inf. Model., 2008, 48(3), 646-658
https://pubs.acs.org/doi/abs/10.1021/ci7004093
"""
import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from cheminformatics_similarity.similarity import get_similarity_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def _upper_triangle_row(i: int, fp_list: list, metric_fn) -> list[float]:
    """
    Compute similarities for one upper-triangle row.

    For row i, compares fp_list[i] against fp_list[i+1:] only.
    This gives exactly the upper-triangle entries for row i, and
    avoids calculating the diagonal and all (j, i) mirror pairs.

    Returns a list of length (N - i - 1).
    Called in parallel by joblib across all rows i in [0, N-2].
    """
    return metric_fn(fp_list[i], fp_list[i + 1:])


def log_identical_smiles_groups(smiles: list[str]) -> None:
    """
    Log metadata about compounds that share identical SMILES strings.

    Databases may contain compounds with different IDs and different
    pIC50 values but the same SMILES due to unresolved stereochemistry.
    These pairs produce Sim = 1.0 and are retained in SALI calculation
    since denominator includes epsilon). The function logs how many
    groups of identical SMILES exist and their sizes.
    """
    counts = Counter(smiles)
    duplicated = {smi: n for smi, n in counts.items() if n > 1}

    if not duplicated:
        logger.info("No duplicate SMILES found in dataset.")
        return

    n_groups = len(duplicated)
    group_sizes = sorted(duplicated.values(), reverse=True)
    n_affected = sum(group_sizes)

    logger.warning(
        f"Found {n_groups} group(s) of identical SMILES affecting {n_affected} compounds.\n"
        f"Group sizes: {group_sizes}.\n"
        f"These pairs will have Sim = 1.0 and may reflect unresolved stereochemistry.\n"
        f"SALI is still computed using denominator epsilon (default 0.001)."
    )


def compute_sali(
    smiles: list[str],
    pic50: np.ndarray,
    fp_list: list,
    metric: str = "tanimoto",
    n_cpus: int = 1,
    sali_epsilon: float = 0.001
) -> pd.DataFrame:
    """
    Compute SALI for all unique upper-triangle pairs (i < j).

    For each row i, BulkTanimotoSimilarity(fp_i, fp_list[i+1:]) is called,
    giving exactly N*(N-1)/2 similarity calculations with no redundancy.
    Avoids building the full N x N matrix and then discarding half i.e.,
    self-similarity diagonal entries and all (j, i) mirror pairs
    
    Rows are dispatched in parallel via joblib.The variable-length row results
    are then flattened and paired with pre-computed (i, j) index arrays.

    The SALI formula used is:
        SALI(i,j) = |pIC50_i - pIC50_j| / (1 - Sim(i,j) + 0.001)

    The epsilon avoids division by zero for identical fingerprints (Sim=1.0)
    
    Note: NaN pIC50 values are excluded before computation. It is the
    caller's responsibility to handle missing values before calling this
    function (e.g. impute, drop, or investigate in a notebook).
    
    Args:
        smiles: list of SMILES strings. Length must equal len(pic50).
        pic50: array of pIC50 values in Molar units.
        fp_list: list of RDKit fingerprint objects. Type must be ExplicitBitVect or UIntSparseIntVect for similarity calculation.
        metric: similarity metric, 'tanimoto' (default) or 'dice'.
        n_cpus: number of parallel workers.
        sali_epsilon: epsilon added to denominator to avoid division by zero at Sim = 1.0.

    Returns:
        pd.DataFrame with Columns
            ['compound_i', 'compound_j', 'SMILES_i', 'SMILES_j',
             'pIC50_i', 'pIC50_j', 'delta_pIC50',
             '{metric}_similarity', 'SALI']
        compound_i/j are the original integer positions in the input lists.
        Output is sorted by SALI descending.
    """
    # validate the inputs
    n = len(smiles)
    if len(pic50) != n or len(fp_list) != n:
        raise ValueError(
            f"smiles ({len(smiles)}), pic50 ({len(pic50)}), and "
            f"fp_list ({len(fp_list)}) must all be the same length."
        )
    metric_fn = get_similarity_fn(metric)
    
    # log any identical smiles
    log_identical_smiles_groups(smiles)

    # exclude any compounds with missing pIC50
    pic50 = np.asarray(pic50, dtype=float)
    smiles_arr = np.asarray(smiles, dtype=object)

    valid_mask = ~np.isnan(pic50)
    n_invalid  = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(
            f"Excluding {n_invalid} compound(s) with NaN pIC50. "
            "Handle missing values in your notebook before calling this function."
        )

    valid_indices = np.where(valid_mask)[0]
    fp_list_valid = [fp_list[i] for i in valid_indices]
    pic50_valid = pic50[valid_mask]
    smiles_valid = smiles_arr[valid_mask]

    n_valid = len(fp_list_valid)
    total_pairs  = n_valid * (n_valid - 1) // 2
    logger.info(
        f"Computing upper-triangle {metric} similarities: "
        f"{n_valid} compounds -> {total_pairs:,} unique pairs "
        f"using {n_cpus} CPU(s)..."
    )

    # Parallel upper-triangle computation
    # Row i returns similarities to compounds [i+1, i+2, ..., N-1].
    # Row N-1 has no upper-triangle partners, so we stop at N-2.
    # row_results[i] has length (N - i - 1):
    # row_results[0] -> N-1 values, row_results[1] -> N-2 values, ..., row_results[N-2] -> 1 value
    row_results = Parallel(n_jobs=n_cpus, backend="loky", batch_size="auto", verbose=5)(
        delayed(_upper_triangle_row)(i, fp_list_valid, metric_fn)
        for i in range(n_valid - 1)
    )

    # Flatten and align row_results with (rows, cols) index arrays
    # np.triu_indices produces the same row-major ordering as our loop
    rows, cols = np.triu_indices(n_valid, k=1)  # both shape (total_pairs,)
    sim_vals = np.concatenate([np.asarray(r) for r in row_results])  # shape (total_pairs,)
    delta_vals = np.abs(pic50_valid[rows] - pic50_valid[cols])

    # SALI formula with epsilon denominator
    sali_vals = delta_vals / (1.0 - sim_vals + sali_epsilon)

    # Map back to original input positions
    orig_i = valid_indices[rows]
    orig_j = valid_indices[cols]

    df_sali = pd.DataFrame({
        "compound_i":           orig_i,
        "compound_j":           orig_j,
        "SMILES_i":             smiles_valid[rows],
        "SMILES_j":             smiles_valid[cols],
        "pIC50_i":              np.round(pic50_valid[rows], 5),
        "pIC50_j":              np.round(pic50_valid[cols], 5),
        "delta_pIC50":          np.round(delta_vals, 5),
        f"{metric}_similarity": np.round(sim_vals, 5),
        "SALI":                 np.round(sali_vals, 5),
    })

    df_sali = df_sali.sort_values("SALI", ascending=False).reset_index(drop=True)
    logger.info(f"Computed {len(df_sali):,} valid SALI pairs.")
    stats_text = (
        f"n pairs = {len(df_sali):,}\n"
        f"mean = {df_sali.SALI.mean():.2f}\n"
        f"median = {df_sali.SALI.median():.2f}\n"
        f"max = {df_sali.SALI.max():.2f}"
    )
    logger.info(stats_text)

    return df_sali


def sort_cliff_pairs_by_potency(df_sali: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder each pair so that compound_i is always the more active compound.

    After compute_sali(), compound_i simply has the lower original index.
    Optionally apply this function to swap compound_i and compound_j 
    (and their associated SMILES and pIC50 columns) so that pIC50_i >= pIC50_j in every row.
    This is purly for easy visual analysis in a notebook.
    The SALI scores, delta_pIC50, and similarity values are symmetric and do not change.

    Args:
        df_sali : pd.DataFrame
        Output of compute_sali(). Must contain columns:
        'compound_i', 'compound_j', 'SMILES_i', 'SMILES_j',
        'pIC50_i', 'pIC50_j'.

    Returns
    pd.DataFrame
        Same shape as input. Rows where pIC50_i < pIC50_j are swapped
        so that pIC50_i >= pIC50_j throughout. Sort order (by SALI) is preserved.
    """
    df = df_sali.copy()
    swap = df["pIC50_i"] < df["pIC50_j"]

    if not swap.any():
        return df

    # swap all paired columns simultaneously
    for col_a, col_b in [
        ("compound_i", "compound_j"),
        ("SMILES_i",   "SMILES_j"),
        ("pIC50_i",    "pIC50_j"),
    ]:
        df.loc[swap, [col_a, col_b]] = df.loc[swap, [col_b, col_a]].values

    logger.info(f"Swapped {swap.sum():,} pairs so that pIC50_i >= pIC50_j throughout.")
    return df


def filter_sali_by_fixed_threshold(
    df_sali: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Return pairs classified as activity cliffs by a fixed SALI threshold.

    Args:
        df_sali: Full SALI results from compute_sali().
        threshold: Minimum SALI value to flag as an activity cliff.
        A common starting point is SALI >= 4.0 for pIC50 data,
        but inspect the distribution plot first to choose empirically
        (see plot_sali_distribution()).

    Returns
        pd.DataFrame: Subset of df_sali where SALI >= threshold.
    """
    df_cliffs = df_sali[df_sali["SALI"] >= threshold].copy()
    logger.info(
        f"Fixed threshold (SALI >= {threshold}): "
        f"{len(df_cliffs):,} activity cliff pairs "
        f"({100 * len(df_cliffs) / max(len(df_sali), 1):.2f}% of all pairs)."
    )
    return df_cliffs


def filter_sali_by_top_n_percent(
    df_sali: pd.DataFrame,
    top_n_percent: float,
) -> pd.DataFrame:
    """
    Return the top N% of pairs by SALI score.

    Args:
    df_sali : pd.DataFrame
        Full SALI results from compute_sali().
    top_n_percent : float
        Percentage of pairs to return, e.g. 5.0 for top 5%.
        Must be between 0 and 100.

    Returns
    pd.DataFrame
        Top N% of df_sali by SALI score (already sorted descending).
    """
    if not (0 < top_n_percent <= 100):
        raise ValueError("top_n_percent must be between 0 and 100.")

    n = max(1, int(np.ceil(len(df_sali) * top_n_percent / 100)))
    df_cliffs = df_sali.head(n).copy()

    logger.info(
        f"Top {top_n_percent}% threshold: {len(df_cliffs):,} activity cliff pairs. "
        f"SALI cutoff at this percentile: {df_cliffs['SALI'].min():.4f}."
    )
    return df_cliffs


def plot_sali_distribution(
    df_sali: pd.DataFrame,
    fixed_threshold: float = None,
    log_scale: bool = False,
):
    """
    Plot the distribution of SALI scores across all pairs.
    Useful for empirically choosing a fixed threshold.

    Displays:
      - Histogram + KDE of all SALI values
      - Vertical dashed lines at 90th, 95th, 99th percentiles
      - Optional vertical line at fixed_threshold

    Args:
    df_sali : pd.DataFrame
        Full SALI results from compute_sali().
    fixed_threshold : float, optional
        If provided, draws a vertical line at this value on the plot.
    log_scale : bool
        If True, plots log10(SALI + 1) on the x-axis. Recommended when
        the distribution is heavily right-skewed. The +1 shift avoids
        log(0) for pairs where delta_pIC50 = 0. Default False.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    sali_vals = df_sali["SALI"]

    if log_scale:
        plot_vals = np.log10(sali_vals + 1)
        xlabel    = "log10(SALI + 1)"
    else:
        plot_vals = sali_vals
        xlabel    = "SALI Score"

    sns.histplot(
        plot_vals,
        bins=100,
        kde=True,
        ax=ax,
        color="steelblue",
        edgecolor="none",
        alpha=0.7,
        label="SALI distribution",
    )

    # Percentile markers to guide threshold selection
    for pct, color, ls in [(90, "orange", "--"), (95, "tomato", "--"), (99, "crimson", ":")]:
        val = np.percentile(plot_vals, pct)
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.5,
                   label=f"{pct}th percentile = {val:.2f}")

    # Fixed threshold line
    if fixed_threshold is not None:
        ax.axvline(fixed_threshold, color="black", linestyle="-", linewidth=2,
                   label=f"Fixed threshold = {fixed_threshold}")

    title = "SALI Score Distribution" + (" (log scale)" if log_scale else "")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()     

    return fig, ax
