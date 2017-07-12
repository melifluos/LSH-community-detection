"""
Generate an LSH table from minhashes

n = 1000.0  # the number of hashes in a signature
b = 500.0  # the number of bands to use - this satisfies thresh ~ (1/b)^(b/n)
r = n / b

The probability that two signatures will hash to one or more identical buckets
is given by 1 - (1-J^r)^b

If one hash function puts two columns in the same bucket, then we regard them as similar

Divide the minhash signatures into b bands each containing r rows

For each band create a hash function with ~ 10^9 buckets (ideally we want no collisions
, but in practice this is impossible)

This means that signatures will only be guaranteed to hash to the same bucket if
they have identical values in one of the bands

Our threshold is quite low and so we need a high b and low r

"""

from __future__ import division
import pandas as pd
import time
import mmh3  # platform independent, robust and fast hash function
import cPickle as pickle
import numpy as np
import argparse


class LSHCandidates:
    """
    A class containing the indices into the signature matrix of accounts returned by
    an LSH query.
    For efficiency we often use the active indices to slice the signature matrix
    before calculating Jaccards. For this reason the class contains functions that
    map from the indices of this reduced signature matrix back to the account indices
    """

    def __init__(self, active_indices):
        """
        :param active_indices: A list of the indices returned by an LSH query
        """
        self.active_indices = active_indices
        self._build_lookups()

    def _build_lookups(self):
        """
        builds lookup dictionaries
        """
        self.active_to_all = {i: self.active_indices[i] for i in range(len(self.active_indices))}
        self.all_to_active = {self.active_indices[i]: i for i in range(len(self.active_indices))}

    def get_active_idx(self, account_index):
        """
        converts from our account indices to an index into the active signatures matrix
        account_index - list or individual account index
        """
        try:
            return [self.all_to_active[x] for x in account_index]  # got an iterable
        except TypeError:
            return self.all_to_active[account_index]  # got an int

    def get_account_idx(self, active_index):
        """
        converts from an index into the active signatures matrix to a account index
        active_index - a list or an individual index into the active signatures
        """
        try:
            return [self.active_to_all[x] for x in active_index]  # got an iterable
        except TypeError:
            return self.active_to_all[active_index]  # got an int


def hash_query_sig(query_sig, band_size):
    """
    produces the hashes to look up for a query account
    """
    query_hash = []
    query_sig = query_sig.astype(np.int64)
    for band_num, col_idx in enumerate(xrange(0, len(query_sig), int(band_size))):
        hash_val = mmh3.hash(query_sig[col_idx:(col_idx + band_size)].tostring())
        # hash_val = mmh3.hash(query_sig[col_idx:(col_idx + band_size)])
        query_hash.append(hash_val)
    return query_hash


def query_account(account_idx, signatures, lsh_table, return_query_id, matches=None):
    """
    runs an LSH query for a single account
    :param account_idx - the index of the account being queried
    :param signatures - A pandas dataframe containing the minhash signature matrix for all accounts
    :param lsh_table - the LSH lookup table
    :param matches - the number of matches. Use none if this is not part of a larger
    :param return_query_id: Boolean. Add the query index to the return values
    query
    """
    # assert isinstance(account_idx, object)
    print 'query account called for index', account_idx, ' of type ', type(account_idx)
    if not matches:
        matches = []

    account_sig = signatures.ix[account_idx, 1:].values

    band_size = len(account_sig) / len(lsh_table)
    query_hash = hash_query_sig(account_sig, band_size)
    for idx, hash_val in enumerate(query_hash):
        try:
            matches += lsh_table[idx][hash_val]
        except KeyError:  # this will happen if no other account hashed to this value in this band
            pass
    # remove duplicates and the original query index
    if return_query_id:
        matches = list(
            set(matches).union(set([account_idx])))  # need to explicitly add the account as any account with no collisions
        # does not get added to the LSH table
    else:
        matches = list(set(matches) - set([account_idx]))

    return matches


def run_query(seeds, signatures, lsh_table, return_query_id=True):
    """
    finds similar accounts for a given account handle
    returns an LSHCandidates object
    seeds - either a dictionary of seed accounts or a single account id
    index_passed - the account index instead of the handle or id was passed. This is faster.
    return_query_id - should the seeds be included in the return value
    :rtype : LSHCandidates
    :param seeds:
    :param signatures:
    :param lsh_table:
    :param index_passed: Did we query with a account's index?
    :param return_query_id:
    """
    start = time.time()
    matches = []

    if isinstance(seeds, dict):
        # get all of the values in the dictionary into a 1D list
        accounts = [account for community in seeds.values() for account in community]
        # accounts = seeds.values()[0]
    else:
        accounts = seeds
    if isinstance(accounts, list):
        for account_idx in accounts:
            print "querying ACCOUNT INDEX", account_idx
            matches = query_account(account_idx, signatures, lsh_table, return_query_id, matches)
    else:  # passed a single account
        account_idx = accounts
        # return all LSH matches for this account
        matches = query_account(account_idx, signatures, lsh_table, return_query_id, matches)
    print len(matches), ' matches identified after ', time.time() - start, ' s'
    print len(matches), ' candidate accounts found'
    if len(matches) == 0:
        return None

    return LSHCandidates(matches)


def hash_single_band(band):
    """
    hashing of a single band of minhash signatures for all accounts
    :param band: a band of minhash signatures for all accounts
    """
    band_hash = {}
    band = band.astype(np.int64)
    # for each account
    for row_idx in range(len(band)):
        hash_val = mmh3.hash(band[row_idx].tostring())
        if hash_val in band_hash:  # add this account to the set of accounts that hash to this value
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
    n_accounts, n_hashes = signatures.shape
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
        nargs='+', default='local_resources/twitter_data.csv', help='the location of the minhash file')
    parser.add_argument(
        'outpath', type=str,
        nargs='+', default='results/community_analysis.csv', help='the location to write data to')

    parser.add_argument(
        '--n_bands', type=int,
        nargs='?', default=500, const=1, help='the number of bands to use in the LSH table')

    args = parser.parse_args()

    inpath = args.inpath[0]
    outpath = args.outpath[0]
    n_bands = args.n_bands
    data = pd.read_csv(inpath, index_col=0)
    signatures = data.values
    build_LSH_table(signatures, outpath, n_bands=n_bands)
