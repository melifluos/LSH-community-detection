# !/usr/bin/env python
"""
This module assesses the quality of our ground truth communities.
We define ground-truth communities as groups of people who all share the same tag 
in our system
author: Ben Chamberlain 26/1/2014
"""

from __future__ import division
import csv
import numpy as np

np.seterr(all='warn')  # To show floating point errors
import pandas as pd
import argparse


def calculate_PR(teleport_set, A, beta=0.85, n_iterations=2):
    """
    calculate the PageRank vector
    :param beta: - the teleport probability, in this case how much probability goes back
    to the seed at each iteration
    :param teleport_set: - a randomly chosed vertex
    :param A: the adjacency matrix for the community
    :param n_iterations: The number of iterations of PR to run, more is more accurate,
    but in practice two is adequate
    :return A numpy array PageRank vector
    """
    # Create seed and label strength vectors
    R = np.ones((A.shape[0], 1), dtype=np.float16)  # Label strength vector
    R_total = np.sum(R, dtype=np.float32)  # Total staring PageRank
    S = np.zeros((A.shape[0], 1), dtype=np.float16)  # Seeds vector
    S[teleport_set] = 1  # Set seeds
    S_num = S.sum()  # Number of seeds
    # propagate label with teleportation
    for dummy_i in range(n_iterations):
        R = beta * np.dot(A, R)
        missing_R = R_total - R.sum()  # Determine the missing PageRank and reinject into the seeds
        R += (missing_R / S_num) * S
    return R.flatten()


def get_cut(PR_vector, degree_vector, A):
    """
    Conducts a Sweep through the PageRank vector
    A Sweep is a commonly used method of extracting a graph cut from a vector.
    Sweeps through the PageRank vector repeatedly recalculating conductance
    and storing the minimum found
    :param PR_vector: The PageRank of each vertex
    :param degree_vector: The degree of each vertex
    :param A: The adjacency matrix
    :return: The minimum value of the conductance found
    """
    # get a degree normalized PR vector
    dn_vector = np.divide(PR_vector, degree_vector)
    # sort in descending order
    sorted_idx = np.argsort(dn_vector)[::-1]

    # TODO: ONLY NEED TO ITERATE OVER NON-ZERO PR VECTORS
    community_size = len(dn_vector)
    total_edge_volume = np.sum(A)  # This sums the edge volume attaced to every node. The double counting is deliberate
    # conduct a Sweep - a te
    min_conductance = 1.0
    for set_size in xrange(2, community_size):
        # get the set with the set_size largest PR vectors
        cut_indices = sorted_idx[0:set_size]
        # get the adjacency matrix for this subgraph
        cut_edges = A[cut_indices, :]
        if 2 * np.sum(cut_edges) > total_edge_volume:
            break  # due to the min in the denominator of conductance, no lower conductance can be found
        internal_edges = cut_edges[:, cut_indices]
        internal_weight = get_internal_weight(internal_edges)
        external_weight = get_external_weight(cut_edges, internal_weight)
        conductance = calculate_conductance(internal_weight, external_weight)
        min_conductance = min(min_conductance, conductance)
    return min_conductance


def get_jaccards(community, all_data):
    """
    a vectorised one versus all Jaccard calculation
    :param community: A pandas DataFrame containing the minhashes for a single community
    :param all_data: A pandas DataFrame containing the minhashes of the dataset
    :return A numpy array of minhash Jaccard estimates of shape(community.shape[0], all_data.shape[0])
    """

    community_size, n_hashes = community.shape
    universe_size, _ = all_data.shape

    print 'calculating jaccard coefficients for', community_size, 'community members against universe of ', universe_size, ' members'

    jaccards = np.zeros((community_size, universe_size))
    community_sigs = community.values
    all_sigs = all_data.values
    for idx in range(community_size):
        comparison_signature = community_sigs[idx, :]

        # tile for broadcasting
        tiled_community = np.tile(comparison_signature, (universe_size, 1))
        # do a vectorize element-wise account1 == account_j for all j
        collisions = all_sigs == tiled_community
        jacc = np.sum(collisions, axis=1) / float(n_hashes)
        jaccards[idx, :] = jacc
        if idx % 100 == 0:
            print 'community member ', idx, ' complete'

    return jaccards


def get_node_list(node_file, col_idx=0, has_header=True):
    """
    takes a csv of ids and returns a list of ids
    col_idx - the column in node_file containing the ids
    """
    retval = []
    with open(node_file, 'r') as f:
        reader = csv.reader(f)
        if has_header:
            reader.next()
        for line in reader:
            retval.append(line[col_idx])
    return retval


def get_internal_weight(internal_edges):
    """
    calculates the weight of internal edges
    internal_edges - the edges weights inside the community
    community_size - number of vertices in the community
    NOTE: people sometimes define internal edge weight by summing the edges of each node independantly, which 
    would give a result twice as large as this function's
    """
    m_s = np.sum(internal_edges) / 2  # matrix is symmetric, only count edges once
    return m_s


def get_external_weight(A, internal_edge_weight):
    """
    calculate the weight of external edges ie. those that 
    connect a node inside of the community with one outside
    A is the adjacency matrix for JUST the nodes in the community
    internal_edge_weight is the sum of the edge weights that connect two nodes in the community    
    """
    total_edge_weight = np.sum(A)  # This is 2*internal_edges + external_edges
    external_edge_weight = total_edge_weight - 2 * internal_edge_weight
    return external_edge_weight


def calculate_separability(m_s, c_s):
    """
    m_s is the total weight of edges between two nodes within the community
    c_s is the total weight of edges where just one node is in the community
    """
    if c_s == 0:
        seperability = m_s
    else:
        seperability = m_s / c_s
    return seperability


def calculate_density(m_s, community_size):
    """
    calculates the density of the communities
    measures the fraction of the edges out of all possible edges 
    that the community contains
    """
    assert community_size > 1, "communities are only defined for two or more nodes"
    max_possible_edge_weight = community_size * (community_size - 1) / 2

    density = m_s / max_possible_edge_weight

    return density


def calculate_cohesiveness(jaccs, n_restarts, community_size):
    """
    calculates the cohesiveness of a community by running sub-community detection 
    and returning the conductance of the lowest conductance sub-community
    n_restarts - the number of randomly selected seeds to try community detection from 
    """
    cohesiveness = 1
    # get the weighted degree of each vertex
    degree = np.sum(jaccs, 0)
    # randomly select the vertices to start with
    starting_vertices = np.random.choice(range(community_size), n_restarts)
    # prepare the adjacency matrix for PageRank by degree normalizing
    adj_mat = np.divide(jaccs, np.tile(degree, (community_size, 1)))

    for vertex in starting_vertices:
        pr = calculate_PR(vertex, adj_mat)
        cohesiveness = min(get_cut(pr, degree, jaccs), cohesiveness)
    return cohesiveness


def calculate_conductance(m_s, c_s, total_edge_weight=float("inf")):
    """
    calculates the conductance of the communities
    m_s is the total internal edge weight (edges between two nodes within the community)
    c_s is the total external edge weight (edges where just one node is in the community)     
    the denominator is actually min(V(s),V(\s)) normally we are only interested in small subsets 
    so V(s) is always smaller, but if in doubt, enter the total edge volume
    """
    if c_s == 0 and m_s == 0:
        conductance = 1
    elif total_edge_weight > 2 * m_s:
        conductance = c_s / (2 * m_s + c_s)
    else:
        conductance = c_s / (2 * (total_edge_weight - m_s) + c_s)
    return conductance


def calculate_clustering_coefficient(internal_edges):
    """
    calculates the clustering coefficient for a weighted graph following the method of Holme et al.
    in P. Holme, S.M. Park, B.J. Kim, and C.R. Edling, Physica A 373, 821 (2007)
    """

    connected_vertices = np.sum(internal_edges, 0) != 0
    print 'removed ', internal_edges.shape[0] - np.sum(connected_vertices), ' vertices'

    internal_edges = internal_edges[connected_vertices, :][:, connected_vertices]

    # cube the matrix, probably a better way to do this
    w3 = internal_edges.dot(internal_edges).dot(internal_edges)

    w_max = np.ones(internal_edges.shape)

    denom = internal_edges.dot(w_max).dot(internal_edges)

    if np.sum(np.diag(denom) == 0) > 0:
        print 'there are ', np.sum(np.diag(denom) == 0), 'zeros in the denominator of the clustering coefficient equation \
        ABOUT TO GET DIVIDE BY ZERO ISSUES!!!'

    # get the clustering coefficients for each node
    clustering_coefficients = np.divide(np.diag(w3), np.diag(denom))

    # we characterize the community by its average clustering coefficient
    return np.mean(clustering_coefficients)


def run_analysis_suite(group, data, generate_graphml=False):
    """ 
    runs the full suite of community analysis metrics for a single community
    """
    # set the number of restarts for cohesiveness - each restart tries to find a different sub-community
    n_iterations = 10

    tag, community = group

    # get the community size, this may not be the same as the number of ids as not all ids are indexed
    community_size = community.shape[0]
    # extract all weighted edges for the community

    jaccs = get_jaccards(community, data)
    indices = community.index.values

    # get just internal edges
    internal_edges = jaccs[:, indices]
    assert np.trace(internal_edges) == community_size, 'error generating jaccards - all self jaccards should be 1'
    # remove the self loops
    np.fill_diagonal(internal_edges, 0)
    # calculate some of the parameters of the metrics
    internal_weight = get_internal_weight(internal_edges)
    external_weight = get_external_weight(jaccs, internal_weight)
    # get the four metrics
    clustering_coefficient = calculate_clustering_coefficient(internal_edges)
    cohesiveness = calculate_cohesiveness(internal_edges, n_iterations, community_size)
    density = calculate_density(internal_weight, community_size)
    separability = calculate_separability(internal_weight, external_weight)
    conductance = calculate_conductance(internal_weight, external_weight)
    # the ratio of external to internal conductance
    conductance_ratio = conductance / cohesiveness
    # return in alphabetic order
    return [community_size, clustering_coefficient, cohesiveness, conductance, conductance_ratio, density, separability]


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

    data = pd.read_csv(args.inpath[0], index_col=0)
    data.index.name = 'community'
    data = data.reset_index()

    full_path = args.outpath[0]
    header = ['community', 'size', 'clustering_coefficient', 'cohesiveness', 'conductance', 'conductance_ratio',
              'density', 'separability']
    with open(full_path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    grouped = data.groupby('community')
    for group in grouped:
        results = run_analysis_suite(group, data, generate_graphml=False)
        print header
        print 'results are ', results
        with open(full_path, 'ab') as f:
            writer = csv.writer(f)
            output = [group[0]] + results
            writer.writerow(output)
