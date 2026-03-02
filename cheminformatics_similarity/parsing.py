from argparse import ArgumentParser


def parse_cli_args(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """
    parser = ArgumentParser(description='Calculates chemical similarity')

    parser.add_argument('--out_file', type=str, default='similarity_matrix.pkl',
                        help='Path to save the similarity matrix as a .pkl file.')

    parser.add_argument('--data_path1', type=str, default='delaney.csv',
                        help='Path to a csv file containing smiles column.')

    parser.add_argument('--data_path2', type=str, default='delaney.csv',
                        help='Path to a csv file containing smiles column to be compared against.')
    
    parser.add_argument('--n_cpus', type=int, default=2,
                        help="Number of CPUs for parallel similarity matrix calculation.")

    parser.add_argument('--n_cpus_featurize', type=int, default=2,
                        help="Number of CPUs for parallel fingerprint generation.")
    
    parser.add_argument("--metric", type=str, default="tanimoto", choices=["tanimoto", "dice"],
                        help="Similarity metric to use (default: tanimoto).")
    
    parser.add_argument("--fp_type", type=str, default="count", choices=["count", "bit"],
                        help="Fingerprint type: count or bit")

    args = parser.parse_args()


    return args
