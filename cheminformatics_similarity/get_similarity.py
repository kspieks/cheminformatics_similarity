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

def get_sim(smi1, smiles_list):
	sim_results = np.zeros(len(smiles_list))
	for i, smi2 in enumerate(smiles_list):
		fp1 = get_RDKit_fp(smi1)
		fp2 = get_RDKit_fp(smi2)

		# todo: explore DataStructs.BulkTanimotoSimilarity() as another way to do this
		sim = DataStructs.TanimotoSimilarity(fp1, fp2)
		sim_results[i] = sim

	return sim_results

def get_similarity_matrix(smiles_list1, smiles_list2, ncpus=1):
	"""
	Returns a matrix of Tanimoto similarities as a numpy array
	of size smiles_list1 x smiles_list2.

	Progress is measured by the number of smiles in smiles_list1.
	"""

	result = Parallel(n_jobs=ncpus, backend='multiprocessing', verbose=5)(delayed(get_sim)(smi, smiles_list2) for smi in smiles_list1)

	similarity_matrix = np.zeros((len(smiles_list1), len(smiles_list2)))
	for i, res in enumerate(result):
		similarity_matrix[i, :] = res

	return similarity_matrix
