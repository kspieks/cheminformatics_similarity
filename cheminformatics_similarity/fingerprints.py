#!/usr/bin/env python3
# encoding: utf-8

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem


fp_gen = AllChem.GetRDKitFPGenerator(minPath=1, maxPath=7, fpSize=2048)

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


def get_fingerprints(smi_list: list, fp_type: str = "count", n_cpus: int = 1) -> list:
    """
    Generate fingerprints in parallel for a list of SMILES strings.

    Args:
        smi_list: List of SMILES strings.
        fp_type:  Fingerprint type. One of 'count' or 'bit'.
        n_cpus:   Number of parallel workers.
    
    Note:
        - bit vectors produce list of rdkit.DataStructs.cDataStructs.ExplicitBitVect
        - count vectors produce list of rdkit.DataStructs.UIntSparseIntVect
    """
    fp_type = fp_type.lower()
    fp_fn = {"count": get_count_RDKit_fp, "bit": get_RDKit_fp_bit}
    if fp_type not in fp_fn:
        raise ValueError(f"Unknown fp_type '{fp_type}'. Choose from: {list(fp_fn)}")
    
    fp_list = Parallel(n_jobs=n_cpus, backend="loky", verbose=5)(
        delayed(fp_fn[fp_type])(smi) for smi in smi_list
    )

    return fp_list
