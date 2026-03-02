#!/usr/bin/env python3
# encoding: utf-8


import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


fp_gen = AllChem.GetRDKitFPGenerator(minPath=1, maxPath=3, fpSize=2048)

def get_RDKit_fp_bit(smi: str):
    """Generate a bit vector fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    # rdkit.DataStructs.cDataStructs.ExplicitBitVect
    return fp_gen.GetFingerprint(mol)


def get_count_RDKit_fp(smi: str):
    """Generate a count vector fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    # rdkit.DataStructs.cDataStructs.UIntSparseIntVect
    return fp_gen.GetCountFingerprint(mol)


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
    metric = metric.lower()
    metric_fn = {
        "tanimoto": _bulk_tanimoto,
        "dice": _bulk_dice,
    }
    if metric not in metric_fn:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(metric_fn)}")

    results = Parallel(n_jobs=ncpus, backend="loky", batch_size="auto", verbose=5)(
        delayed(metric_fn[metric])(fp, fp_list2)
        for fp in fp_list1
    )
    similarity_matrix = np.array(results)

    return similarity_matrix
