"""
Generates minhashes from an edge list. The intensive computation is handled in cython
"""
import joblib
import numpy as np
from numpy.random.mtrand import RandomState
from minhash_numpy import calculate_minhashes
import pandas as pd


def read_edges(x_path, y_path):
    labels = pd.read_csv(y_path, header=None, sep=' ')
    edges = pd.read_csv(x_path, header=None, sep=' ', dtype=np.uint64)
    return edges, labels


def generate_hashes(x_path, y_path, out_path, num_hashes):
    hash_a, hash_b = multiply_shift_params(num_hashes=num_hashes)
    edges, labels = read_edges(x_path, y_path)
    edge_arr = edges.values.copy(order='c')
    num_vertices = labels.shape[0]
    signatures = np.zeros((num_vertices, num_hashes), dtype=np.uint32)
    # Set the signatures to the maximum 32bit unsigned integer value so that any value lower will replace it.
    signatures[:] = np.iinfo(np.uint32).max
    # this writes to the signatures array
    print hash_a.shape
    print hash_b.shape
    print signatures.dtype
    print edge_arr.dtype
    hash_a = np.expand_dims(hash_a, axis=0)
    hash_b = np.expand_dims(hash_b, axis=0)
    calculate_minhashes(edge_arr, signatures, hash_a, hash_b, num_hashes)
    df = pd.DataFrame(index=labels[1], data=signatures)
    df.to_csv(out_path)


def multiply_shift_params(num_hashes=1000):
    hash_a_pkl = '../../resources/hash_a_{}.pkl'.format(num_hashes)
    hash_b_pkl = '../../resources/hash_b_{}.pkl'.format(num_hashes)
    hash_a = None
    hash_b = None
    try:
        print "Previous hash parameters found and will be used."
        hash_a = joblib.load(hash_a_pkl)
        hash_b = joblib.load(hash_b_pkl)
    except IOError:
        print "No pickle file for hash values found in resources directory. Creating new hash values."

    if hash_a is None and hash_b is None:
        # Remove the seed to vary the randomised generated hashes between successive executions
        rand = RandomState(seed=0)

        # From: http://en.wikipedia.org/wiki/Universal_hashing
        # a = random positive odd integer < 2^word bits
        # Select three times the number of hashes to make sure we get enough odd numbers.
        a_nums = rand.uniform(0, np.iinfo(np.uint64).max, num_hashes * 3)

        a_integers = a_nums.astype(np.uint64) + 1
        hash_a = a_integers[np.mod(a_integers, 2) != 0][:num_hashes]
        # b = random non-negative integer b < 2^w-M (64 bits in machine word, 2^31.48=3e9, w-M = 2^33)
        hash_b = rand.random_integers(0, 2 ** 33, num_hashes)
        hash_b = hash_b.astype(np.uint64)

        joblib.dump(hash_a, hash_a_pkl)
        joblib.dump(hash_b, hash_b_pkl)

    return hash_a, hash_b


if __name__ == '__main__':
    # LSH.LSHCandidates([1, 2])
    x_path = '../../local_resources/email_data/email-Eu-core.txt'
    y_path = '../../local_resources/email_data/labels.txt'
    out_path = '../../local_resources/email_data/signatures.txt'
    generate_hashes(x_path, y_path, out_path, num_hashes=100)
