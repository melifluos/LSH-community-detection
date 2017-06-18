"""
Run the experiments on the ground truth communities
"""

from __future__ import division
import numpy as np
# from scipy.sparse import lil_matrix, coo_matrix
import os, csv, sys
# import multiprocessing as mp
# import pdb
import pandas as pd
import LSH
# pdb.set_trace()
import cPickle as pickle
from sets import Set
from time import gmtime, strftime, time  # , sleep
from scipy.spatial.distance import pdist, squareform
import random as rand


class CommunityDetector:
    """
    A class to detect communities of accounts using seed expansion techniques
    """

    def __init__(self, data, outfolder, lsh_path):
        """

        :param use_lsh:
        :return:
        """
        self.signatures = data
        self.community_sizes = data.groupby('community').size()
        self.active_signatures = None
        self.outfolder = outfolder
        self.load_lsh_table(lsh_path)
        self.lsh_candidates = None
        self.used_idx = {}

    def calculate_initial_average_similarity(self, seeds):
        """
        calculates the average similarity of every account with the
        seeds
        :param seeds: A default dictionary of the form {community: [seeds]}
        :returns A pandas DataFrame of shape (n_communities, n_active_accounts) indexed by community name
        """
        n_accounts, n_hashes = self.active_signatures.shape
        n_communities = len(seeds)

        # get the average similarity between a community and all other accounts
        average_similarities = np.zeros(shape=(n_communities, n_accounts))
        index = []
        for set_num, candidates in enumerate(seeds.iteritems()):
            # store the individual similarities for each account
            name, accounts = candidates
            index.append(name)
            similarities = np.zeros(shape=(len(accounts), n_accounts))
            for similarity_idx, account_idx in enumerate(accounts):
                # convert to an index into just the active signatures
                row_idx = self.lsh_candidates.get_active_idx(account_idx)
                similarities[similarity_idx, :] = self.get_account_similarities(self.active_signatures,
                                                                                self.active_signatures[row_idx, :])

            average_similarities[set_num, :] = similarities.mean(0)
        df = pd.DataFrame(data=average_similarities, index=index)
        return df

    def output_best_initial_averages(self, account_similarities, seeds,
                                     n_seeds, n_accounts, interval, file_name):
        """
        Extract accounts with the highest average Jaccards with the input seeds
        :param account_similarities:
        :param seeds:
        :param n_seeds:
        :param n_accounts:
        :param interval:
        :param file_name:
        :return:
        """
        # get an array of account identifiers in similarity order for every community
        sorted_vals = np.argsort(-account_similarities.values)
        sorted_idx = pd.DataFrame(data=sorted_vals, index=account_similarities.index)

        # n_communities = account_similarities.shape[0]
        with open(self.outfolder + '/' + file_name, 'ab') as f:
            writer = csv.writer(f)
            for name, val in seeds.iteritems():
                n_members = self.community_sizes[name]
                out_line = []
                # community = int(key) - 1
                hit_count = 0
                total_recall = 0
                for idx, account_idx in enumerate(sorted_idx.ix[name, :]):
                    # account_id = self.index_to_id(account_idx)
                    # jacc = account_similarities[community,account_idx]
                    try:
                        result_line = self.signatures.ix[account_idx, :]
                        # result_line = tags_df.loc[tags_df['NetworkID'] == int(account_id)]
                    except TypeError:
                        print account_idx, ' of type ', type(account_idx), 'caused type error'
                        # print account_id, ' of type ', type(account_id), 'caused type error'
                        raise
                    if name == str(result_line['community']):
                        hit_count += 1
                    if (idx + 1) % interval == 0:  # record data at this point
                        # how much of the entire set did we get
                        total_recall = (hit_count - n_seeds) / float(n_members - n_seeds)
                        out_line.append(format(total_recall, '.4f'))

                    # stop when we have enough accounts
                    if idx == n_accounts:
                        writer.writerow(out_line)
                        break

                if idx < n_accounts:  # this happens with bad communities when there are fewer LSH candidates than community members
                    n_cols = len(xrange(interval, n_accounts, interval))
                    for idx in range(n_cols - len(out_line)):
                        out_line.append(format(total_recall,
                                               '.4f'))  # recall won't improve as no more candidates
                    writer.writerow(out_line)

    def update_account_similarities(self, community_similarities, row_idx, community, community_size):
        """
        Adds a new account to the average similarities
        :param community_similarities: A pandas dataframe with index community_names and shape (n_communities, n_active_accounts)
        :param row_idx: The index of this account into the active signature matrix
        :param community: String. The name of the current community.
        :param community_size: Int. The size of the community called <community> ie. the seeds plus any additions
        :return: None
        """
        # row_idx = self.id_to_index(new_account)
        # find the similarity between this account and all others
        new_account_similarities = self.get_account_similarities(self.active_signatures,
                                                                 self.active_signatures[row_idx, :])
        # perform the streaming mean update: mu(t+1) = (n*mu(t)+x)/(n+1)
        try:
            community_similarities.ix[community, :] = (community_size * community_similarities.ix[community, :].values +
                                                       new_account_similarities) / (community_size + 1)
        except ValueError:
            pass

    def increment_communities(self, account_similarities, seeds):
        """
        Find the most similar account not already in each community and
        add them to it.
        :param account_similarities: A pandas dataframe of account similarities indexed by the community names. Shape
        (n_communities, n_active_accounts)
        :param seeds: A default dictionary of the seeds of the form {community_name:[acc_idx1, acc_idx2,...],...}
        :return: None
        """
        # n_communities = len(seeds)
        # get an array of account identifiers in similarity order for every community
        temp = np.argsort(-account_similarities.values)
        sorted_idx = pd.DataFrame(data=temp, index=account_similarities.index)

        # for each community, try the accounts in decreasing similarity order
        for community in seeds.keys():
            if not community in self.used_idx:
                # self.used_ids[community_idx] = Set([])
                self.used_idx[community] = Set([])
            col_idx = 0
            while True:
                # get the index of the high jaccard account into the active indices
                try:
                    active_idx = sorted_idx.ix[community, col_idx]
                except IndexError:
                    print 'no accounts left to add to the community'
                    raise
                # convert from an index into the active accounts to a account id
                account_idx = self.lsh_candidates.get_account_idx(active_idx)
                # account_id = self.index_to_id(account_idx)

                # if account_id not in self.used_ids[community_idx] and account_id:
                if account_idx not in self.used_idx[community] and account_idx:
                    # add the new account
                    # get the size of the community for the community update equation
                    community_size = len(seeds[community])
                    # self.update_account_similarities(account_similarities, account_id, community_idx,
                    #                                  community_size)
                    self.update_account_similarities(account_similarities, active_idx, community,
                                                     community_size)
                    # get the Jaccard for this new account with the community
                    jacc = account_similarities.ix[community, active_idx]
                    # add the new account to the community
                    # account_handle = self.account_lookup.id(account_id)['handle']
                    # seeds[str(community_idx + 1)].append((int(account_id), account_handle, jacc))
                    seeds[community].append((account_idx, jacc))
                    # self.used_ids[community_idx].add(account_id)
                    self.used_idx[community].add(account_idx)

                    # move to the next community
                    break
                else:  # check the next account
                    col_idx += 1

    def get_account_similarities(self, all_signatures, test_signatures):
        """
        Gets the estimated Jaccard for all accounts
        with all communities where a community is the union of its member accounts
        :param all_signatures: Numpy array of shape (n_active_signatures, n_hashes). Minhash signatures of every active account
        :param test_signatures: Numpy array of shape (n_hashes, n_test_accounts). Signatures of the accounts we wish to compare with all_signatures. n_test_accounts is usually 1
        :return Numpy array of shape
        """
        n_accounts, signature_size = all_signatures.shape
        try:
            n_communities, signature_size = test_signatures.shape
        except ValueError:  # we used a single account
            n_communities = 1

        similarities = np.zeros(shape=(n_communities, n_accounts))

        for row_idx in range(n_communities):
            # for each account find the % of hashes that agree with the seeds
            try:
                similarities[row_idx, :] = np.sum((all_signatures == test_signatures[row_idx, :]), axis=1) / float(
                    signature_size)
            except IndexError:  # only one community / account
                similarities[row_idx, :] = np.sum((all_signatures == test_signatures), axis=1) / float(signature_size)

        return similarities

    def output_results(self, communities):
        """
        outputs the communities to file
        :param communities: default dictionary of type community_name:[(acc_idx1, jacc1), (acc_idx2, jacc2), ...], ...].
        The seeds plus all additions to the communities
        """
        truth = self.signatures.index
        with open(self.outfolder + '/community_output.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['community', 'detected community', 'jaccard'])
            for key, val in communities.iteritems():
                tags_df['community'] = key
                for account in val:
                    try:
                        outline = [key, truth[account[0]], account[1]]
                    except IndexError:  # it was a seed, so doesn't have a jaccard. Put in a jaccard of 1.0
                        outline = [key, truth[account], 1.0]

                    # try:
                    #     result_line = tags_df.loc[tags_df['NetworkID'] == int(account[0])]
                    # except TypeError:
                    #     print account[0], ' of type ', type(account[0])
                    #     raise
                    #     # result_line.loc['community'] = key
                    # try:
                    #     result_line = result_line.iloc[0]
                    # except IndexError:
                    #     pass
                    # out_line = list(result_line)
                    # try:  # add the jaccard with the community what the account was added
                    #     out_line.append(account[2])
                    # except IndexError:  # They were a seed
                    #     out_line.append(1)
                    writer.writerow(outline)

    def calculate_recall(self, communities, n_seeds, n_accounts,
                         interval=None):
        """
        Calculates the recall against the ground truth community for a specific
        target community size.
        interval - if specified will calculate recall at various intervals
        """
        truth = self.signatures.index
        with open(self.outfolder + '/minrank.csv', 'ab') as f:
            writer = csv.writer(f)
            for key, val in communities.iteritems():
                name, n_members = self.community_sizes[key]
                hit_count = 0
                results = []
                for idx, account in enumerate(val):
                    try:
                        true_community = truth[val[0]]
                    except IndexError:
                        true_community = truth[val]
                    if true_community == key:
                        hit_count += 1

                    # result_line = tags_df.loc[tags_df['NetworkID'] == int(account[0])]
                    # if name in str(result_line['Tag']):
                    #     hit_count += 1
                    if (idx + 1) % interval == 0:
                        # how much of the entire set did we get
                        total_recall = (hit_count - n_seeds) / float(n_members - n_seeds)
                        results.append(format(total_recall, '.4f'))
                if idx < n_accounts:  # this happens with bad communities when there are fewer LSH candidates than community members
                    n_cols = len(xrange(interval, n_accounts, interval))
                    for new_idx in range(n_cols - len(results)):
                        results.append(format(total_recall, '.4f'))  # recall won't improve as no more candidates
                # what is our current percentage success rate
                # set_recall = (hit_count - n_seeds) / float(target_size)
                writer.writerow(results)

    def pageRank(self, seeds, k_iterations=3, beta=0.9):
        """
        Calculate the personalised PageRank for each active vertex
        :param seeds: A default dictionary of community_name:[seed_idx1, seed_idx2,..], ... The original seeds
        :param k_iterations: Int. Number of steps for the random walks
        :param beta: Float. 1 - teleport probability
        :return: A Pandas Dataframe shape (n_communities, n_active_accounts) indexed by community names. The personalized PageRank of every active account with respect to the seeds of each community
        """
        # get the full similarity matrix using jaccard_sim = 1 - jaccard_dist
        A = 1 - squareform(pdist(self.active_signatures, 'jaccard'))
        # seed_index_list = []
        # get all of the values in the dictionary into a 1D list
        index = []
        PR = np.zeros(shape=(len(seeds), A.shape[0]))
        for idx, (community, accounts) in enumerate(seeds.iteritems()):
            index.append(community)
            seed_index_list = [self.lsh_candidates.get_active_idx(account) for account in accounts]
            # seed_index_list = [self.lsh_candidates.get_active_idx(account) for community in seeds.values() for account in
            #                    community]
            # for set_num, accounts in seed_ids.iteritems():
            #     # store the individual similarities for each account
            #     for account_idx, account_id in enumerate(accounts):
            #         try:
            #             seed_index_list.append(self.id_to_index(account_id[0]))
            #         except IndexError:
            #             print account_id, 'NOT IN INDEX'
            #         except KeyError:
            #             print account_id, 'NOT IN LSH CANDIDATES - PROBABLY BECAUSE THERE IS NO T-FILE'
            # Create seed and rank vectors
            R = np.ones((A.shape[0], 1))  # Rank vector
            R_total = R.sum()  # Total starting PageRank
            S = np.zeros((A.shape[0], 1))  # Seeds vector
            S[seed_index_list] = 1  # Set seeds
            S_num = S.sum()  # Number of seeds

            # Normalise
            print "{} Normalising matrix for pageRank".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            connection_sum = A.sum(axis=0)
            np.divide(A, connection_sum, out=A)

            # Start label propagation
            t0 = time()
            for i in range(1, k_iterations + 1):
                print "\n{} Starting itteration {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), i)

                # propagate labels with teleportation
                R = beta * A.dot(R)
                missing_R = R_total - R.sum()  # Determine the missing PageRank and reinject into the seeds
                R += (missing_R / S_num) * S
            PR[idx, :] = R.T

        df = pd.DataFrame(data=PR, index=index)

        return df

    def run_community_detection(self, seeds, n_accounts=50, n_seeds=5, result_interval=10,
                                runtime_file=None):
        """
        runs community detection from seeds
        :param seeds: - default dictionary of type community_name:[acc_idx1, acc_idx2,...], ...].
        The seeds to start the community detection with. The process appends to the seeds
        :param community sizes - the total size of the ground truthed communities
        :param n_accounts - the maximum number of accounts to grow the community to
        :param n_seeds: The number of seeds to use
        :param result_interval - the number of accounts that are added to the seeds between each reading of the recall
        :param runtime_file: write the runtime of these methods to file
        :return default dictionary of type community_name:[acc_idx1, acc_idx2,...], ...].
        The seeds to start the community detection with. The process appends to the seeds
        """
        start_time = time()

        # Use the locality sensitive hashing table to conduct an initial nearest neighbours search
        if not isinstance(self.lsh_table, list):
            print 'loading lsh lookup table'
            self.load_lsh_table()
        print 'running lsh query'
        print seeds
        self.lsh_candidates = LSH.run_query(seeds, self.signatures, self.lsh_table, return_query_id=True)
        # reduce the signatures matrix to only work with nearby accounts
        self.active_signatures = self.signatures.ix[self.lsh_candidates.active_indices, 1:].values
        n_candidates = len(self.active_signatures)
        if n_candidates < n_accounts:
            print "not all community members are active. Will only consider ", n_candidates, ' of the ', n_accounts, ' accounts'
            n_additions = n_candidates
        else:
            n_additions = n_accounts
            # implement a new lookup

        # find the jaccard distance to all non-seeds averaged over the seeds
        ast0 = time()
        account_similarities = self.calculate_initial_average_similarity(seeds)
        avg_sim_time = time() - ast0
        self.output_best_initial_averages(account_similarities, seeds,
                                          n_seeds, n_accounts, result_interval, file_name='initial_avgs.csv')
        prt0 = time()
        R = self.pageRank(seeds)
        pr_time = time() - prt0
        self.output_best_initial_averages(R, seeds, n_seeds, n_accounts,
                                          result_interval, file_name='pagerank.csv')

        srt0 = time()
        for idx in range(n_additions):
            # Adds the next most similar account to each group of seeds and updates the average distance from the community members to all other accounts
            self.increment_communities(account_similarities, seeds)
            # record the recall every
            if (idx + 1) % 10 == 0:
                print idx + 1, 'accounts added'
        sim_rank_time = time() - srt0

        if runtime_file:
            writer = csv.writer(runtime_file)
            community = self.outfolder.rsplit('/', 1)[-1]
            writer.writerow(['page_rank', community, pr_time])
            writer.writerow(['min_rank', community, sim_rank_time])
            writer.writerow(['avg_sim_time', community, avg_sim_time])

        print 'added', n_accounts, 'into each of', len(seeds), ' communities in ', time() - start_time, 'seconds'
        return seeds

    def load_lsh_table(self, path):
        """
        loads the locality sensitive hash lookup table into memory
        :param path: the path to the LSH table
        """
        start = time()
        print 'reading pickled hash table'
        self.lsh_table = pickle.load(open(path, 'rb'))
        print 'table read into memory in time ', time() - start, ' s'

    def run_experimentation(self, n_seeds, group, random_seeds, result_interval, runtime_file):
        """
        performs a series of community detection runs using different seeds
        and measure recall
        :param n_seeds: The number of seeds to start with. Default is 5
        :param result_interval: the intervals in number of accounts to snap the recall at. Default is 10
        """
        name = group[0]
        hashes = group[1]
        n_accounts, n_hashes = hashes.shape

        with open(self.outfolder + '/minrank.csv', 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_accounts, result_interval)
            writer.writerow(cols)

        with open(self.outfolder + '/initial_avgs.csv', 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_accounts, result_interval)
            writer.writerow(cols)

        with open(self.outfolder + '/pagerank.csv', 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_accounts, result_interval)
            writer.writerow(cols)

        start_time = time()

        for idx, rdm_seed in enumerate(random_seeds):
            # generate Twitter account indices to seed the local community detection
            # seeds = self.generate_seeds(rdm_seed, n_seeds=5)  #  experimental value
            seeds = self.generate_seeds(rdm_seed, n_seeds=1)  # debug value

            print seeds

            communities = self.run_community_detection(seeds,
                                                       n_accounts=n_accounts, n_seeds=n_seeds,
                                                       result_interval=result_interval, runtime_file=runtime_file)
            print 'completed for ', n_accounts, 'accounts in ', time() - start_time

            # print 'community shape ',communities
            self.output_results(communities)
            self.calculate_recall(communities, n_seeds, n_accounts,
                                  result_interval)

        print 'experimentation completed for ', len(random_seeds), ' random restarts in ', time() - start_time

    def generate_seeds(self, rdm_seed, n_seeds=5):
        """
        generates seed indices
        :param random_seed: seed for the random number generator
        :param n_seeds: The number of accounts to return
        :return A python dictionary of the form community_name: [seed indices]]
        """
        rand.seed(rdm_seed)

        grouped = self.signatures.groupby('community')
        sample = grouped.apply(lambda x: x.sample(n_seeds, random_state=rdm_seed))
        import collections
        seeds = collections.defaultdict(list)
        for elem in sample.index:
            seeds[elem[0]].append(elem[1])

        return seeds


if __name__ == '__main__':
    n_seeds = 30  # The number of seeds to start with
    result_interval = 10  # the intervals in number of accounts to snap the recall at
    # random_seeds = [451235, 35631241, 2315, 346213456, 134]  #  experimental choices of seeds

    random_seeds = [451235]

    inpath = '../../local_resources/twitter_data.csv'
    outfolder = '../../results'
    lsh_path = '../../results/hash_table.pkl'

    data = pd.read_csv(inpath, index_col=0)
    data.index.name = 'community'
    data = data.reset_index()
    community_detector = CommunityDetector(data, outfolder, lsh_path=lsh_path)
    with open('../../results/runtimes_' + str(n_seeds) + '.csv', 'wb') as runtime_file:
        writer = csv.writer(runtime_file)
        writer.writerow(['community', 'method', 'runtime'])
    with open('../../results/runtimes_' + str(n_seeds) + '.csv', 'ab') as runtime_file:
        grouped = data.groupby('community')
        community_size = grouped.size()
        for group in grouped:
            community_detector.run_experimentation(n_seeds, group, random_seeds, result_interval, runtime_file)
