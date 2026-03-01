#!/usr/bin/env python3
# encoding: utf-8

import pickle as pkl

import pandas as pd
from joblib import Parallel, delayed

from cheminformatics_similarity.get_similarity import get_RDKit_fp, get_similarity_matrix
from cheminformatics_similarity.parsing import parse_cli_args


def main():
    """
    The main executable function.
    """

    args = parse_cli_args()

    df1 = pd.read_csv(args.data_path1)
    df2 = pd.read_csv(args.data_path2)

    # get list of rdkit.DataStructs.UIntSparseIntVect
    print("Generating fingerprints...")
    fp_list1 = Parallel(n_jobs=args.n_cpus_featurize, backend="loky", verbose=5)(
        delayed(get_RDKit_fp)(smi) for smi in df1.smiles.tolist()
    )
    fp_list2 = Parallel(n_jobs=args.n_cpus_featurize, backend="loky", verbose=5)(
        delayed(get_RDKit_fp)(smi) for smi in df2.smiles.tolist()
    )

    print(f'fp_list1 has {len(fp_list1)} smiles')
    print(f'fp_list2 has {len(fp_list2)} smiles')
    print("Computing similarity matrix")
    print('Progress is measured by the number of molecules in fp_list1...')

    similarity_matrix = get_similarity_matrix(fp_list1, fp_list2, ncpus=args.n_cpus)
    print(f'similarity_matrix.shape: {similarity_matrix.shape}')

    with open(args.out_file, 'wb') as f:
        pkl.dump(similarity_matrix, f)

if __name__ == '__main__':
    main()
