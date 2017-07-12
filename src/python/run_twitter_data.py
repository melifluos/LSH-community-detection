"""
A script to run the Twitter data from the experimental evaluation of https://arxiv.org/abs/1601.03958.
Added for reproducibility - the code is not to be executed end-to-end in production
"""

from generate_hashes import generate_hashes
import pandas as pd
from LSH import build_LSH_table
from run_experiments import run_email_experiments

if __name__ == '__main__':
    x_path = '../../resources/email-Eu-core.txt'
    y_path = '../../resources/labels.txt'
    sig_path = '../../local_resources/email_signatures.txt'
    lsh_path = '../../local_resources/email_hash_table.pkl'
    out_folder = '../../results'
    generate_hashes(x_path, y_path, sig_path, num_hashes=100)
    data = pd.read_csv(sig_path, index_col=0)
    signatures = data.values
    build_LSH_table(signatures, lsh_path, n_bands=50)
    run_email_experiments(sig_path, out_folder, lsh_path)