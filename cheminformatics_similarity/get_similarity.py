#!/usr/bin/env python3
# encoding: utf-8


import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


fp_gen = AllChem.GetRDKitFPGenerator(minPath=1, maxPath=3, fpSize=2048)

def get_RDKit_fp(smi):
	"""Generate a count fingerprint for a given SMILES string."""
	mol = Chem.MolFromSmiles(smi)
	return fp_gen.GetCountFingerprint(mol)


def _bulk_tanimoto(fp: object, fp_list: list) -> list:
    """
    Thin wrapper around DataStructs.BulkTanimotoSimilarity required because
    Boost.Python functions cannot be pickled by joblib's loky backend.
    """
    return DataStructs.BulkTanimotoSimilarity(fp, fp_list)


def get_similarity_matrix(
    fp_list1: list,
    fp_list2: list,
    ncpus: int = 1,
) -> np.ndarray:
	"""
	Returns a matrix of Tanimoto similarities as a numpy array
	of shape (len(fp_list1), len(fp_list2)).

	Args:
        fp_list1: Query fingerprints (rows of the output matrix).
        fp_list2: Reference fingerprints (columns of the output matrix).
        ncpus:    Number of parallel workers.

	Progress is measured by the number of smiles in fp_list1.
	"""
	results = Parallel(n_jobs=ncpus, backend="loky", batch_size="auto", verbose=5)(
		delayed(_bulk_tanimoto)(fp, fp_list2)
		for fp in fp_list1
	)
	similarity_matrix = np.array(results)

	return similarity_matrix
