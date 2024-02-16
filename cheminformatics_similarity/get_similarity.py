#!/usr/bin/env python3
# encoding: utf-8


from joblib import Parallel, delayed
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


fp_gen = AllChem.GetRDKitFPGenerator(minPath=1, maxPath=3, fpSize=2048)

def get_RDKit_fp(smi):
	mol = Chem.MolFromSmiles(smi)
	return fp_gen.GetCountFingerprint(mol)

def get_sim(fp, fp_list):
	
	similarities = DataStructs.BulkTanimotoSimilarity(fp, fp_list)

	return similarities


def get_similarity_matrix(fp_list1, fp_list2, ncpus=1):
	"""
	Returns a matrix of Tanimoto similarities as a numpy array
	of size fp_list1 x fp_list2.

	Progress is measured by the number of smiles in fp_list1.
	"""

	result = Parallel(n_jobs=ncpus, backend='multiprocessing', verbose=5)(delayed(get_sim)(fp, fp_list2) for fp in fp_list1)

	similarity_matrix = np.zeros((len(fp_list1), len(fp_list2)))
	for i, res in enumerate(result):
		similarity_matrix[i, :] = res

	return similarity_matrix
