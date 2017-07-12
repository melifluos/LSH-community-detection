"""
A script to run the end-to-end process for the public email data. Added for reproducibility - the code is not
to be executed end-to-end in production
"""

from generate_hashes import generate_hashes
import pandas as pd
from LSH import build_LSH_table

if __name__ == '__main__':
    x_path = '../../resources/email-Eu-core.txt'
    y_path = '../../resources/labels.txt'
    sig_path = '../../local_resources/signatures.txt'
    lsh_path = '../../local_resources/hash_table.pkl'
    generate_hashes(x_path, y_path, sig_path, num_hashes=100)
    data = pd.read_csv(sig_path, index_col=0)
    signatures = data.values
    build_LSH_table(signatures, lsh_path, n_bands=50)
