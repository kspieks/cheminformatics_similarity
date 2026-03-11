#!/usr/bin/env python3
# encoding: utf-8


import numpy as np
from joblib import Parallel, delayed
from rdkit import DataStructs


def _bulk_tanimoto(fp, fp_list):
    """
    Thin wrapper around DataStructs.BulkTanimotoSimilarity required because
    Boost.Python functions cannot be pickled by joblib's loky backend.

    T(A, B) = c / (a + b - c)
    c = number of bits in common (intersection of a and b)
    """
    return DataStructs.BulkTanimotoSimilarity(fp, fp_list)


def _bulk_dice(fp, fp_list):
    """
    Thin wrapper around DataStructs.BulkDiceSimilarity required because
    Boost.Python functions cannot be pickled by joblib's loky backend.

    D(A, B) = 2c / (a + b)
    c = number of bits in common (intersection of a and b)

    Since Dice normalizes by the average number of bits rather than the union,
    Dice values are always greater than or equal to Tanimoto values, which
    can compress the dynamic range at high similarity.
    """
    return DataStructs.BulkDiceSimilarity(fp, fp_list)


_METRIC_FNS = {
    "tanimoto": _bulk_tanimoto,
    "dice":     _bulk_dice,
}

def get_similarity_fn(metric: str = "tanimoto"):
    """Return the bulk similarity callable for the given metric."""
    metric = metric.lower()
    if metric not in _METRIC_FNS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(_METRIC_FNS)}")

    metric_fn = _METRIC_FNS[metric]
    return metric_fn


def get_similarity_matrix(
    fp_list1: list,
    fp_list2: list,
    ncpus: int = 1,
    metric: str = "tanimoto",
) -> np.ndarray:
    """
    Returns a similarity matrix as a numpy array of shape (len(fp_list1), len(fp_list2)).
    The list of fingerprints can be either count or bit fingerprints (but should be consistent).

    Args:
        fp_list1: Query fingerprints (rows of the output matrix).
        fp_list2: Reference fingerprints (columns of the output matrix).
        ncpus:    Number of parallel workers.
        metric:   Similarity metric to use. One of 'tanimoto' or 'dice'.

    Progress is measured by the number of smiles in fp_list1.
    """
    metric_fn = get_similarity_fn(metric=metric)

    results = Parallel(n_jobs=ncpus, backend="loky", batch_size="auto", verbose=5)(
        delayed(metric_fn)(fp, fp_list2)
        for fp in fp_list1
    )
    similarity_matrix = np.array(results)

    return similarity_matrix
