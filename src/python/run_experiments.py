"""
Run the experiments on the ground truth communities
"""

from __future__ import division
import numpy as np
import csv
import pandas as pd
import LSH
import cPickle as pickle
from sets import Set
from time import gmtime, strftime, time
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
        community = seeds.keys()[0]
        sorted_idx = account_similarities.sort_values(community, axis=1, ascending=False)

        with open(self.outfolder + '/' + file_name, 'ab') as f:
            writer = csv.writer(f)
            for name, val in seeds.iteritems():
                n_members = self.community_sizes[name]
                out_line = []
                hit_count = 0
                total_recall = 0
                for idx, active_idx in enumerate(sorted_idx.columns.values):
                    account_idx = self.lsh_candidates.get_account_idx(active_idx)
                    try:
                        result_line = self.signatures.ix[account_idx, :]
                    except TypeError:
                        print account_idx, ' of type ', type(account_idx), 'caused type error'
                        raise
                    if (name == str(result_line['community'])) and (account_idx not in val):
                        hit_count += 1
                    if (idx + 1) % interval == 0:  # record data at this point
                        # how much of the entire set did we get
                        total_recall = hit_count / float(n_members - n_seeds)
                        out_line.append(format(total_recall, '.4f'))
                    # stop when we have enough accounts
                    if (idx + 1) == n_accounts:
                        writer.writerow(out_line)
                        break
                if sorted_idx.shape[1] < n_accounts:
                    # this happens with bad communities when there are fewer LSH candidates than community members
                    n_cols = len(xrange(interval, n_accounts, interval))
                    for idx in range(n_cols - len(out_line)):
                        out_line.append(format(total_recall,
                                               '.4f'))  # recall won't improve as no more candidates
                    writer.writerow(out_line)

    def output_best_initial_averages1(self, account_similarities, seeds,
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

        with open(self.outfolder + '/' + file_name, 'ab') as f:
            writer = csv.writer(f)
            for name, val in seeds.iteritems():
                n_members = self.community_sizes[name]
                out_line = []
                hit_count = 0
                total_recall = 0
                for idx, active_idx in enumerate(sorted_idx.ix[name, :]):
                    account_idx = self.lsh_candidates.get_account_idx(active_idx)
                    try:
                        result_line = self.signatures.ix[account_idx, :]
                    except TypeError:
                        print account_idx, ' of type ', type(account_idx), 'caused type error'
                        raise
                    if (name == str(result_line['community'])) and (account_idx not in val):
                        hit_count += 1
                    if (idx + 1) % interval == 0:  # record data at this point
                        # how much of the entire set did we get
                        total_recall = hit_count / float(n_members - n_seeds)
                        out_line.append(format(total_recall, '.4f'))
                    # stop when we have enough accounts
                    if idx == n_accounts:
                        writer.writerow(out_line)
                        break

                if len(
                        out_line) < n_accounts:  # this happens with bad communities when there are fewer LSH candidates than community members
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
        indices = community_similarities.columns.values  # this table is transposed
        # find the similarity between this account and all others
        new_account_similarities = self.get_account_similarities(self.active_signatures,
                                                                 self.active_signatures[row_idx, :])
        # perform the streaming mean update: mu(t+1) = (n*mu(t)+x)/(n+1)
        community_similarities.ix[community, :] = (community_size * community_similarities.ix[community, :].values +
                                                   new_account_similarities[:, indices]) / (community_size + 1)

    def increment_communities(self, account_similarities, seeds):
        """
        Find the most similar account not already in each community and
        add them to it.
        :param account_similarities: A pandas dataframe of account similarities indexed by the community names. Shape
        (n_communities, n_active_accounts)
        :param seeds: A default dictionary of the seeds of the form {community_name:[acc_idx1, acc_idx2,...],...}
        :return: None. The function alters seeds inplace.
        """
        community = seeds.keys()[0]
        sorted_idx = account_similarities.sort_values(community, axis=1, ascending=False)
        active_idx = sorted_idx.columns[0]

        account_idx = self.lsh_candidates.get_account_idx(active_idx)

        community_size = len(seeds[community])
        self.update_account_similarities(account_similarities, active_idx, community,
                                         community_size)
        # get the Jaccard for this new account with the community
        jacc = account_similarities.ix[community, active_idx]
        # add the new account to the community
        seeds[community].append((account_idx, jacc))
        # remove this account once added
        account_similarities.drop(active_idx, axis=1, inplace=True)

    def get_account_similarities(self, all_signatures, test_signatures):
        """
        Gets the estimated Jaccard for all accounts
        with all communities where a community is the union of its member accounts
        :param all_signatures: Numpy array of shape (n_active_signatures, n_hashes). Minhash signatures of every active account
        :param test_signatures: Numpy array of shape (n_hashes, n_test_accounts). Signatures of the accounts we wish to compare with all_signatures. n_test_accounts is usually 1
        :return Numpy array of shape (n_communities, n_active_signatures)
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
        return: A pandas Dataframe of shape (len(communities.values()), 2) and columns ['detected_community', 'jaccard']
        """
        truth = self.signatures.community
        community = communities.keys()[0]
        name = community.replace(" ", "_")
        pairs = communities.values()[0]
        # add jaccard of 1.0 to the seeds and look up truth communities of accounts
        full_pairs = [(truth[val[0]], val[1]) if hasattr(val, '__iter__') else (truth[val], 1.0) for val in pairs]
        # convert to numpy array
        data = np.array(zip(*full_pairs))
        df = pd.DataFrame(columns=['detected_community', 'jaccard'], index=range(data.shape[1]), data=data.T)
        df.index.name = community
        path = self.outfolder + '/' + name + '_community_output.csv'
        df.to_csv(path, index=None)
        return df

        # with open(self.outfolder + '/' + name + '_community_output.csv', 'wb') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(
        #         ['community', 'detected community', 'jaccard'])
        #     for key, val in communities.iteritems():
        #         for account in val:
        #             try:
        #                 outline = [key, truth[account[0]], account[1]]
        #             except IndexError:  # it was a seed, so doesn't have a jaccard. Put in a jaccard of 1.0
        #                 outline = [key, truth[account], 1.0]
        #             writer.writerow(outline)

    def calculate_recall(self, communities, n_seeds, n_accounts,
                         interval=None):
        """
        Calculates the recall against the ground truth community for a specific
        target community size.
        :param communities: default dictionary of type community_name:[(acc_idx1, jacc1), (acc_idx2, jacc2), ...], ...].
        The seeds plus all additions to the communities
        :param n_seeds: Int. The number of seeds to initialise each community with
        :param n_accounts: Int. The number of accounts in the dataset
        :param interval: Int. if specified will calculate recall at various intervals
        :return:
        """
        name = communities.index.name
        minrank_path = self.outfolder + "/" + name.replace(" ", "_") + '_minrank.csv'
        minrank_result = communities.detected_community.values
        # remove seeds
        minrank_result = minrank_result[n_seeds:]
        n_none_seeds = n_accounts - n_seeds
        recall = []
        hit_count = 0
        for idx, true_community in enumerate(minrank_result):
            if true_community == name:
                hit_count += 1
            if (idx + 1) % interval == 0:
                total_recall = hit_count / float(n_none_seeds)
                recall.append(total_recall)
        if n_none_seeds > len(minrank_result):
            last_val = recall[-1]
            for dummy_i in range(0, (n_none_seeds - len(minrank_result)) // interval):
                recall.append(last_val)  # recall can't get any higher as LSH didn't return any more results

            with open(minrank_path, 'ab') as f:
                writer = csv.writer(f)
            writer.writerow(recall)
            #     for key, val in communities.iteritems():
            #         n_members = self.community_sizes[key]
            #         hit_count = 0
            #         results = []
            #         total_recall = 0
            #         for idx, account in enumerate(val):
            #             try:
            #                 true_community = truth[val[0]]
            #             except TypeError:
            #                 true_community = truth[val]
            #             if true_community == key:
            #                 hit_count += 1
            #             if (idx + 1) % interval == 0:
            #                 # how much of the entire set did we get
            #                 total_recall = (hit_count - n_seeds) / float(n_ - n_seeds)
            #                 results.append(format(total_recall, '.4f'))
            #         # this happens with bad communities when there are fewer LSH candidates than community members
            #         if idx < n_accounts:
            #             n_cols = len(xrange(interval, n_accounts, interval))
            #             for new_idx in range(n_cols - len(results)):
            #                 results.append(format(total_recall, '.4f'))  # recall won't improve as no more candidates
            #         writer.writerow(results)

    def calculate_recall1(self, communities, n_seeds, n_accounts,
                          interval=None):
        """
        Calculates the recall against the ground truth community for a specific
        target community size.
        :param communities: default dictionary of type community_name:[(acc_idx1, jacc1), (acc_idx2, jacc2), ...], ...].
        The seeds plus all additions to the communities
        :param n_seeds: Int. The number of seeds to initialise each community with
        :param n_accounts: Int. The number of accounts in the dataset
        :param interval: Int. if specified will calculate recall at various intervals
        :return:
        """
        truth = self.signatures.community
        name = communities.keys()[0]
        minrank_path = self.outfolder + "/" + name.replace(" ", "_") + '_minrank.csv'
        with open(minrank_path, 'ab') as f:
            writer = csv.writer(f)
            for key, val in communities.iteritems():
                n_members = self.community_sizes[key]
                hit_count = 0
                results = []
                total_recall = 0
                for idx, account in enumerate(val):
                    try:
                        true_community = truth[val[0]]
                    except TypeError:
                        true_community = truth[val]
                    if true_community == key:
                        hit_count += 1
                    if (idx + 1) % interval == 0:
                        # how much of the entire set did we get
                        total_recall = (hit_count - n_seeds) / float(n_members - n_seeds)
                        results.append(format(total_recall, '.4f'))
                # this happens with bad communities when there are fewer LSH candidates than community members
                if idx < n_accounts:
                    n_cols = len(xrange(interval, n_accounts, interval))
                    for new_idx in range(n_cols - len(results)):
                        results.append(format(total_recall, '.4f'))  # recall won't improve as no more candidates
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
            for i in range(1, k_iterations + 1):
                print "\n{} Starting itteration {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), i)

                # propagate labels with teleportation
                R = beta * A.dot(R)
                missing_R = R_total - R.sum()  # Determine the missing PageRank and reinject into the seeds
                R += (missing_R / S_num) * S
            PR[idx, :] = R.T

        df = pd.DataFrame(data=PR, index=index)

        return df

    def remove_seeds_from_similarities(self, similarities, seeds):
        """
        remove the seed accounts from the account similarities so that they are not added by the minrank
        :param account_similarities: A pandas dataframe of account similarities indexed by the community names. Shape
        (n_communities, n_active_accounts)
        :param seeds: A default dictionary of the seeds of the form {community_name:[acc_idx1, acc_idx2,...],...}
        :return: None. The function alters seeds inplace.
        """
        seed_indices = []
        for seed in seeds.values()[0]:
            seed_indices.append(self.lsh_candidates.get_active_idx(seed))

        similarities.drop(seed_indices, axis=1, inplace=True)

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
        community = seeds.keys()[0]
        name = community.replace(" ", "_")

        # Use the locality sensitive hashing table to conduct an initial nearest neighbours search
        if not isinstance(self.lsh_table, list):
            print 'loading lsh lookup table'
            self.load_lsh_table()
        print 'running lsh query'

        self.lsh_candidates = LSH.run_query(seeds, self.signatures, self.lsh_table, return_query_id=True)
        # reduce the signatures matrix to only work with nearby accounts
        self.active_signatures = self.signatures.ix[self.lsh_candidates.active_indices, 1:].values
        n_candidates = len(self.active_signatures)
        if n_candidates < n_accounts:
            print "not all community members are active. Will only consider ", n_candidates, ' of the ', n_accounts, ' accounts'
            n_additions = n_candidates - n_seeds
        else:
            n_additions = n_accounts - n_seeds

        # find the jaccard distance to all non-seeds averaged over the seeds
        ast0 = time()
        account_similarities = self.calculate_initial_average_similarity(seeds)
        # remove the seeds before incrementing
        self.remove_seeds_from_similarities(account_similarities, seeds)
        avg_sim_time = time() - ast0
        self.output_best_initial_averages(account_similarities, seeds,
                                          n_seeds, n_accounts - n_seeds, result_interval,
                                          file_name=name + '_initial_avgs.csv')
        prt0 = time()
        R = self.pageRank(seeds)
        pr_time = time() - prt0
        self.output_best_initial_averages(R, seeds, n_seeds, n_accounts - n_seeds,
                                          result_interval, file_name=name + '_page_rank.csv')
        srt0 = time()
        for idx in range(n_additions):
            # Adds the next most similar account to each group of seeds and updates the
            # average distance from the community members to all other accounts
            self.increment_communities(account_similarities, seeds)
            # record the recall every
            if (idx + 1) % 10 == 0:
                print idx + 1, 'accounts added'
        sim_rank_time = time() - srt0
        # reset the used IDs after the loop
        self.used_idx = {}

        if runtime_file:
            writer = csv.writer(runtime_file)
            community = seeds.keys()[0]
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
        n_hashes -= 1  # don't count the community name column

        minrank_path = self.outfolder + "/" + name.replace(" ", "_") + '_minrank.csv'
        with open(minrank_path, 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_accounts - n_seeds, result_interval)
            writer.writerow(cols)

        initial_averages_path = self.outfolder + "/" + name.replace(" ", "_") + '_initial_avgs.csv'
        with open(initial_averages_path, 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_accounts - n_seeds, result_interval)
            writer.writerow(cols)

        page_rank_path = self.outfolder + "/" + name.replace(" ", "_") + '_page_rank.csv'
        with open(page_rank_path, 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_accounts - n_seeds, result_interval)
            writer.writerow(cols)

        start_time = time()

        for idx, rdm_seed in enumerate(random_seeds):
            # generate Twitter account indices to seed the local community detection
            seeds = self.generate_seeds(rdm_seed, group, n_seeds=n_seeds)  # debug value

            print seeds

            communities = self.run_community_detection(seeds,
                                                       n_accounts=n_accounts, n_seeds=n_seeds,
                                                       result_interval=result_interval, runtime_file=runtime_file)
            print 'completed for ', n_accounts, 'accounts in ', time() - start_time
            # write minrank discovered community labels to file for analysis
            minrank_community_df = self.output_results(communities)
            # generate minrank output
            self.calculate_recall(minrank_community_df, n_seeds, n_accounts,
                                  result_interval)

        print 'experimentation completed for ', len(random_seeds), ' random restarts in ', time() - start_time

    def generate_seeds(self, rdm_seed, group, n_seeds=5):
        """
        generates seed indices
        :param random_seed: seed for the random number generator
        :param n_seeds: The number of accounts to return
        :return A python dictionary of the form community_name: [seed indices]]
        """
        np.random.seed(rdm_seed)
        community_name = group[0]
        indices = group[1].index.values
        sample = np.random.choice(indices, n_seeds, replace=False)
        import collections
        seeds = collections.defaultdict(list)
        for elem in sample:
            seeds[community_name].append(elem)

        return seeds


if __name__ == '__main__':
    start_time = time()
    n_seeds = 30  # The number of seeds to start with. Experimental value
    result_interval = 10  # the intervals in number of accounts to snap the recall at
    random_seeds = [451235, 35631241, 2315, 346213456, 134]  # experimental choices of seeds

    # random_seeds = [451235]

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
            if group[0] == 'cosmetics':
                community_detector.run_experimentation(n_seeds, group, random_seeds, result_interval, runtime_file)

    print 'All experiments for ', len(random_seeds), ' random restarts in ', time() - start_time
