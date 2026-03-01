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
                        help='Path to save predictions. The similarity matrix will be stored as a .pkl file.')

    parser.add_argument('--data_path1', type=str, default='delaney.csv',
                        help='Path to a csv file containing SMILES.')
    parser.add_argument('--data_path2', type=str, default='delaney.csv',
                        help='Path to a csv file containing SMILES to be compared to.')
    
    parser.add_argument('--n_cpus', type=int, default=2,
                        help='Number of CPUs to use in parallel when calculating the similarity matrix.')
    parser.add_argument('--n_cpus_featurize', type=int, default=2,
                        help='Number of CPUs to use in parallel when creating feature vectors.')
    
    args = parser.parse_args()


    return args
