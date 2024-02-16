#!/usr/bin/env python3
# encoding: utf-8
import pickle as pkl

import pandas as pd
from pandarallel import pandarallel

from cheminformatics_similarity.parsing import parse_training_command_line_arguments
from cheminformatics_similarity.get_similarity import get_RDKit_fp, get_similarity_matrix

def main():
    """
    The main executable function.
    """

    args = parse_training_command_line_arguments()
    pandarallel.initialize(nb_workers=args.n_cpus_featurize, progress_bar=True)

    df1 = pd.read_csv(args.data_path1)
    df2 = pd.read_csv(args.data_path2)

    # get list of rdkit.DataStructs.UIntSparseIntVect
    df1['fp'] = df1.smiles.parallel_apply(get_RDKit_fp)
    df2['fp'] = df2.smiles.parallel_apply(get_RDKit_fp)

    fp_list1 = list(df1.fp.values)
    fp_list2 = list(df2.fp.values)
    print(f'fp_list1 has {len(fp_list1)} smiles')
    print(f'fp_list2 has {len(fp_list2)} smiles')
    print('Progress is measured by the number of molecules in fp_list1')

    similarity_matrix = get_similarity_matrix(fp_list1, fp_list2, ncpus=args.n_cpus)
    print(f'similarity_matrix.shape: {similarity_matrix.shape}')

    with open(args.out_file, 'wb') as f:
        pkl.dump(similarity_matrix, f)

if __name__ == '__main__':
    main()
