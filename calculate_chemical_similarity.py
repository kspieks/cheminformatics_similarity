#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pandas as pd

from cheminformatics_similarity.fingerprints import get_fingerprints
from cheminformatics_similarity.parsing import parse_cli_args
from cheminformatics_similarity.similarity import get_similarity_matrix


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

    # compress and save as .npz
    # the keyword 'similarity' becomes the key used for loading
    # to load it back later:
    # loaded = np.load(args.out_file)
    # similarity_matrix = loaded['similarity']
    np.savez_compressed(args.out_file, similarity=similarity_matrix)

if __name__ == '__main__':
    main()
