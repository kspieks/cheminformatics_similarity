#!/usr/bin/env python3
# encoding: utf-8

import pickle as pkl

import pandas as pd
from joblib import Parallel, delayed

from cheminformatics_similarity.get_similarity import (get_count_RDKit_fp,
                                                       get_RDKit_fp_bit,
                                                       get_similarity_matrix)
from cheminformatics_similarity.parsing import parse_cli_args


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


def main():
    """
    The main executable function.
    """

    args = parse_cli_args()

    df1 = pd.read_csv(args.data_path1)
    df2 = pd.read_csv(args.data_path2)

    print(f"Generating fingerprints for {args.data_path1}...")
    fp_list1 = get_fingerprints(df1.smiles.tolist(), fp_type=args.fp_type, n_cpus=args.n_cpus_featurize)

    print(f"Generating fingerprints for {args.data_path2}...")
    fp_list2 = get_fingerprints(df2.smiles.tolist(), fp_type=args.fp_type, n_cpus=args.n_cpus_featurize)

    print(f'fp_list1 has {len(fp_list1)} smiles')
    print(f'fp_list2 has {len(fp_list2)} smiles')
    print("Computing similarity matrix")
    print('Progress is measured by the number of molecules in fp_list1...')

    similarity_matrix = get_similarity_matrix(
        fp_list1, fp_list2, ncpus=args.n_cpus, metric=args.metric
    )
    print(f'similarity_matrix.shape: {similarity_matrix.shape}')

    with open(args.out_file, 'wb') as f:
        pkl.dump(similarity_matrix, f)

if __name__ == '__main__':
    main()
