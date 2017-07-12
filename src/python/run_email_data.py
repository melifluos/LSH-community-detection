"""
A script to run the end-to-end process for the public email data. Added for reproducibility - the code is not
to be executed end-to-end in production
"""

from generate_hashes import generate_hashes

if __name__ == '__main__':
    x_path = '../../resources/email-Eu-core.txt'
    y_path = '../../resources/labels.txt'
    out_path = '../../local_resources/email_data/signatures.txt'
    generate_hashes(x_path, y_path, out_path, num_hashes=100)