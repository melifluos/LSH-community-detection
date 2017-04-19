"""
Generate an LSH table from minhashes
"""

from __future__ import division
import pandas as pd
import time
import mmh3  # platform independent, robust and fast hash function
import cPickle as pickle
import argparse

def hash_single_band(band):
    """
    hashing of a single band of minhash signatures for all stars
    band - a band of minhash signatures for all stars
    """
    band_hash = {}
    # for each star
    for row_idx in range(len(band)):
        hash_val = mmh3.hash(band[row_idx].tostring())
        if hash_val in band_hash:  # add this star to the set of stars that hash to this value
            band_hash[hash_val].append(row_idx)
        else:
            band_hash[hash_val] = [row_idx]
    # We only need to store collisions
    collisions = {key: val for key, val in band_hash.items() if len(val) > 1}
    return collisions


def build_LSH_table(signatures, outpath, n_bands=500):
    """
    builds the Locality Sensitive Hashing (LSH) lookup table to be used by
    all locality sensitive queries
    :param n_bands: is the number of bands to hash to. The higher this is the lower
    the Jaccard threshold to be considered 'local'
    :param signatures: the minhash signatures
    :param outpath: where to write the lsh_table to
    :return The hash table list[list[]]
    """
    start = time.time()
    hash_table = []
    n_stars, n_hashes = signatures.shape
    band_size = n_hashes / n_bands
    for band_num, col_idx in enumerate(xrange(0, n_hashes, int(band_size))):
        # need to efficiently find any duplicate hashes. These are the only ones we care about
        hashes = hash_single_band(signatures[:, col_idx:(col_idx + band_size)])
        hash_table.append(hashes)
        print band_num, ' bands complete in time ', time.time() - start, ' s'

    pickle.dump(hash_table, open(outpath, 'wb'), protocol=2)

    return hash_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate community features',
                                     epilog='features are based just on the communities and not all of Twitter')
    parser.add_argument(
        'inpath', type=str,
        nargs='+', default='local_resources/plos_one_data.csv', help='the location of the minhash file')
    parser.add_argument(
        'outpath', type=str,
        nargs='+', default='results/community_analysis.csv', help='the location to write data to')

    args = parser.parse_args()

    print args.inpath[0]
    print args.outpath[0]

    inpath = 'local_resources/plos_one_data.csv'
    inpath = args.inpath[0]
    outpath = 'results/hash_table.pkl'
    outpath = args.outpath[0]
    data = pd.read_csv(inpath, index_col=0)
    signatures = data.values
    build_LSH_table(signatures, outpath)
