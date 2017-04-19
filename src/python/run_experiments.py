"""
Run the experiments on the ground truth communities
"""

from __future__ import division
import numpy as np
# from scipy.sparse import lil_matrix, coo_matrix
import os, csv, sys
# import multiprocessing as mp
# import pdb
import select_seeds
import pandas as pd
import LSH
# pdb.set_trace()
import cPickle as pickle
from sets import Set
from time import gmtime, strftime, time  # , sleep
from scipy.spatial.distance import pdist, squareform

# from utilities_python.numpy_parts import dot_parts - JOSH WROTE A PIECEWISE MATRIX MULTIPLICATION ROUTINE

package_directory = os.path.dirname(os.path.abspath(__file__))
TAG_FILE_PATH = os.path.join(package_directory, 'resources', 'all_handles_tag3.csv')

# sys.path.append(os.environ['GIT_ANALYTICS']+'/utilities_python')
sys.path.append(os.path.join(package_directory, '..', 'utilities_python'))

from influencer_index.src import star_index


# read in the input seeds file

class CommunityDetector():
    """
    A class to detect communities of stars using seed expansion techniques
    """

    def __init__(self, data, outfolder, use_lsh):
        """

        :param use_lsh:
        :return:
        """
        self.signatures = data
        self.active_signatures = None
        self.outfolder = outfolder
        if use_lsh:
            self.load_lsh_table()
        else:
            self.lsh_table = None
        self.use_lsh = use_lsh
        self.lsh_candidates = None


    def calculate_initial_average_similarity(self, communities):
        """
        calculates the average similarity of every star with the
        members of the communities
        """
        n_stars, n_hashes = self.active_signatures.shape
        n_communities = len(communities)

        # get the average similarity between a community and all other stars
        average_similarities = np.zeros(shape=(n_communities, n_stars))

        for set_num, stars in communities.iteritems():
            # store the individual similarities for each star
            similarities = np.zeros(shape=(len(stars), n_stars))
            for star_idx, star_id in enumerate(stars):
                try:
                    row_idx = self.id_to_index(star_id[0])
                except IndexError:
                    print star_id, 'NOT IN INDEX'
                    continue
                except KeyError:
                    print star_id, 'NOT IN LSH CANDIDATES - PROBABLY BECAUSE THERE WAS NO T-FILE'
                    continue
                # find the similarity between this star and all others
                try:
                    similarities[star_idx, :] = self.get_star_similarities(self.active_signatures,
                                                                           self.active_signatures[row_idx, :])
                except IndexError:
                    pass
            average_similarities[int(set_num) - 1, :] = similarities.mean(0)
        return average_similarities

    def output_best_initial_averages(self, star_similarities, seeds, tags_df, community_folder, community_sizes,
                                     n_seeds, n_stars, interval, file_name):
        """
        Extract stars with the highest average Jaccards with the input seeds
        :param star_similarities:
        :param seeds:
        :param tags_df:
        :param community_folder:
        :param community_sizes:
        :param n_seeds:
        :param n_stars:
        :param interval:
        :param file_name:
        :return:
        """
        # get an array of star identifiers in similarity order for every community
        sorted_idx = np.argsort(-star_similarities)
        # n_communities = star_similarities.shape[0]
        with open(community_folder + '/' + file_name, 'ab') as f:
            writer = csv.writer(f)
            for key, val in seeds.iteritems():
                name, n_members = community_sizes[key]
                out_line = []
                community = int(key) - 1
                hit_count = 0
                total_recall = 0
                for idx, star_idx in enumerate(sorted_idx[community, :]):
                    star_id = self.index_to_id(star_idx)
                    # jacc = star_similarities[community,star_idx]
                    try:
                        result_line = tags_df.loc[tags_df['NetworkID'] == int(star_id)]
                    except TypeError:
                        print star_id, ' of type ', type(star_id), 'caused type error'
                        raise
                    if name in str(result_line['Tag']):
                        hit_count += 1
                    if (idx + 1) % interval == 0:  # record data at this point
                        # how much of the entire set did we get
                        total_recall = (hit_count - n_seeds) / float(n_members - n_seeds)
                        out_line.append(format(total_recall, '.4f'))

                    # stop when we have enough stars
                    if idx == n_stars:
                        writer.writerow(out_line)
                        break

                if idx < n_stars:  # this happens with bad communities when there are fewer LSH candidates than community members
                    n_cols = len(xrange(interval, n_stars, interval))
                    for idx in range(n_cols - len(out_line)):
                        out_line.append(format(total_recall,
                                               '.4f'))  # recall won't improve as no more candidates
                    writer.writerow(out_line)

    def update_star_similarities(self, community_similarities, new_star, community, community_size):
        """
        adds a new star to the average similarities
        """
        row_idx = self.id_to_index(new_star)
        # find the similarity between this star and all others
        new_star_similarities = self.get_star_similarities(self.active_signatures, self.active_signatures[row_idx, :])
        # perform the streaming mean update: mu(t+1) = (n*mu(t)+x)/(n+1)
        try:
            community_similarities[community, :] = (community_size * community_similarities[community,
                                                                     :] + new_star_similarities) / (community_size + 1)
        except ValueError:
            pass

    def increment_communities(self, star_similarities, communities, excluded_set=None, community_hashes=None):
        """
        Find the most similar star not already in each community and
        add them to it.
        """
        n_communities = len(communities)
        # get an array of star identifiers in similarity order for every community
        sorted_idx = np.argsort(-star_similarities)

        # for each community, try the stars in decreasing similarity order
        for community_idx in range(n_communities):
            if not community_idx in self.used_ids:
                self.used_ids[community_idx] = Set([])
            col_idx = 0
            while True:
                # get the index of the high jaccard star into the active indices
                try:
                    star_idx = sorted_idx[community_idx, col_idx]
                except IndexError:
                    print 'no stars left to add to the community'
                    raise
                # convert from an index into the active stars to a star id
                star_id = self.index_to_id(star_idx)

                if star_id not in self.used_ids[
                    community_idx] and star_id not in excluded_set:  # check if this star is already a member of the community
                    # add the new star
                    if community_hashes:  # if we're representing the whole community with a single signature, then update the signature
                        community_hashes[community_idx, :] = np.minimum(community_hashes[community_idx, :],
                                                                        self.active_signatures[star_idx, :])
                    else:  # update the similarities to include the new star
                        # get the size of the community for the community update equation
                        community_size = len(communities[str(community_idx + 1)])
                        self.update_star_similarities(star_similarities, star_id, community_idx, community_size)
                        # get the Jaccard for this new star with the community
                    jacc = star_similarities[community_idx, star_idx]
                    # add the new star to the community
                    star_handle = self.star_lookup.id(star_id)['handle']
                    communities[str(community_idx + 1)].append((int(star_id), star_handle, jacc))
                    self.used_ids[community_idx].add(star_id)

                    # move to the next community
                    break
                else:  # check the next star
                    col_idx += 1

    def get_star_similarities(self, all_signatures, test_signatures):
        """
        Function that gets the estimated Jaccard for all stars
        with all communities where a community is the union of its member stars
        ARGS:
        all_signatures - minhash signatures of every star
        test_signatures - signatures of the stars we wish to compare with all other stars
        """
        n_stars, signature_size = all_signatures.shape
        try:
            n_communities, signature_size = test_signatures.shape
        except ValueError:  # we used a single star
            n_communities = 1

        similarities = np.zeros(shape=(n_communities, n_stars))

        for row_idx in range(n_communities):
            # for each star find the % of hashes that agree with the seeds
            try:
                similarities[row_idx, :] = np.sum((all_signatures == test_signatures[row_idx, :]), axis=1) / float(
                    signature_size)
            except IndexError:  # only one community / star
                similarities[row_idx, :] = np.sum((all_signatures == test_signatures), axis=1) / float(signature_size)

        return similarities

    def output_results(self, tags_df, communities, community_folder):
        """
        outputs the communities to file with the full information
        found in the tags file for each star
        tags_df - the data frame version of the input csv file
        communities - the communities we have detected
        """
        with open(community_folder + '/community_output.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['community', 'NetworkID', 'Handle', 'Followers', 'Reported', 'Ctry1', 'Ctry2', 'Tag', 'Toptag',
                 'newhandle', 'jacc'])
            for key, val in communities.iteritems():
                tags_df['community'] = key
                for star in val:
                    try:
                        result_line = tags_df.loc[tags_df['NetworkID'] == int(star[0])]
                    except TypeError:
                        print star[0], ' of type ', type(star[0])
                        raise
                        # result_line.loc['community'] = key
                    try:
                        result_line = result_line.iloc[0]
                    except IndexError:
                        pass
                    out_line = list(result_line)
                    try:  # add the jaccard with the community what the star was added
                        out_line.append(star[2])
                    except IndexError:  # They were a seed
                        out_line.append(1)
                    writer.writerow(out_line)
        return

    def calculate_recall(self, tags_df, communities, community_folder, community_sizes, n_seeds, n_stars,
                         interval=None):
        """
        Calculates the recall against the ground truth community for a specific
        target community size.
        interval - if specified will calculate recall at various intervals
        """
        with open(community_folder + '/minrank.csv', 'ab') as f:
            writer = csv.writer(f)
            for key, val in communities.iteritems():
                name, n_members = community_sizes[key]
                hit_count = 0
                results = []
                for idx, star in enumerate(val):
                    result_line = tags_df.loc[tags_df['NetworkID'] == int(star[0])]
                    if name in str(result_line['Tag']):
                        hit_count += 1
                    if (idx + 1) % interval == 0:
                        # how much of the entire set did we get
                        total_recall = (hit_count - n_seeds) / float(n_members - n_seeds)
                        results.append(format(total_recall, '.4f'))
                if idx < n_stars:  # this happens with bad communities when there are fewer LSH candidates than community members
                    n_cols = len(xrange(interval, n_stars, interval))
                    for new_idx in range(n_cols - len(results)):
                        results.append(format(total_recall, '.4f'))  # recall won't improve as no more candidates
                # what is our current percentage success rate
                # set_recall = (hit_count - n_seeds) / float(target_size)
                writer.writerow(results)

    def get_excluded_set(self):
        """
        Removes the stars that we currently think have broken signatures
        # we start the signatures by getting all of the stars currently in our artirix db and assigning them
        the maximum integer. If the star is not present in our T-files ie. we haven't yet gathered its followers, then at
        # the end of the signature building process it will still have the max int and so we need to remove
        # any stars where this is the case
        """
        excluded_set = set()

        # Exclude Stars which the signatures havn't been generated
        star_ids_list = self.star_lookup.id_list()
        star_ids_list = np.array(star_ids_list)
        max_int = np.iinfo(np.uint32).max
        # get the list of all signatures with the max int
        signatures_inf = (self.signatures[:, 0] == max_int)
        excluded_list = star_ids_list[signatures_inf]
        excluded_set = excluded_set.union(excluded_list)

        # Exclude bad signatures
        # bad_signatures_list = self.detect_bad_signatures(2, star_ids_list)
        # excluded_set = excluded_set.union(bad_signatures_list)

        return excluded_set

    def pageRank(self, star_similarities, seed_ids, k_iterations=3, print_full_info=False, beta=0.9):
        '''
        A: Normalised jaccard matrix times by 1000 (int representation)
        seeds_index_list: List of indes of the seeds
        k_iterations: the number of iteration for pagerank to perform
        '''
        # get the full similarity matrix using jaccard_sim = 1 - jaccard_dist
        A = 1 - squareform(pdist(self.active_signatures, 'jaccard'))
        seed_index_list = []
        for set_num, stars in seed_ids.iteritems():
            # store the individual similarities for each star
            for star_idx, star_id in enumerate(stars):
                try:
                    seed_index_list.append(self.id_to_index(star_id[0]))
                except IndexError:
                    print star_id, 'NOT IN INDEX'
                except KeyError:
                    print star_id, 'NOT IN LSH CANDIDATES - PROBABLY BECAUSE THERE IS NO T-FILE'
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
            # BELOW IS THE DOT PARTS FUNCTION THAT JOSH WROTE, WHICH I NO LONGER HAVE
            # R = beta * dot_parts(A, R, a_parts=4, b_parts=1)  # split this up as the full matrix overflows memory

            # R = R / 10000                       # Renormalise back to 0 to 1
            missing_R = R_total - R.sum()  # Determine the missing PageRank and reinject into the seeds
            R += (missing_R / S_num) * S

            if print_full_info == True:
                # Find top Stars
                top_star_index = np.argsort(-R[:, 0])[0:10]
                top_star_ids = self.index_to_id(top_star_index)
                top_star = self.star_lookup.id(top_star_ids)['handle']
                top_star_strength = R[top_star_index, 0]

                print "Average label strength: {}".format(R.mean())
                print "Top Stars with strengths:"
                print list(top_star)
                print list(top_star_strength)

                dt = time() - t0
                print "Itteration took {}s".format(round(dt, 1))
                t0 = time()

        return R.T

    def run_community_detection(self, community_folder, tags, seeds, excluded_set=None, community_sizes=None,
                                n_stars=50, n_seeds=5,
                                min_seed_followers=1e4, use_union=False, max_followers=1e7, generate_seeds=True,
                                result_interval=10, seed_file=None, runtime_file=None):
        """
        runs community detection from seeds
        :param community_folder - the folder containing the seed_communities.csv file
        :param tags: dataframe containing the community tags
        :param seeds: - the seeds to start the community detection with. The process appends to the seeds
        :param excluded_set: Stars not to use
        :param community sizes - the total size of the ground truthed communities
        :param n_stars - the maximum number of stars to grow the community to
        :param n_seeds: The number of seeds to use
        :param min_seed_followers - seeds need to have more than this number of followers
        :param use_union - represent the community as the union of all of the stars fans
        :param max_followers - don't allow any stars larger than this value to form communities
        :param generate_seeds - if true randomly select seeds from a given tag class. Otherwise read them from file
        :param result_interval - the number of stars that are added to the seeds between each reading of the recall
        :param seed_file - required if generate_seeds = False
        :param runtime_file: write the runtime of these methods to file
        """
        start_time = time()

        # Exclude big Stars
        if max_followers:
            bad_ids = tags[tags['Followers'] >= max_followers]
            excluded_set = excluded_set.union(set(bad_ids['NetworkID']))

        if self.use_lsh:  # Use the locality sensitive hashing table to conduct an initial nearest neighbours search
            if not isinstance(self.lsh_table, list):
                print 'loading lsh lookup table'
                self.load_lsh_table()
            print 'running lsh query'
            print seeds
            self.lsh_candidates = LSH.run_query(seeds, self.signatures, self.lsh_table, self.star_lookup,
                                                index_passed=False, return_query_id=True)
            # reduce the signatures matrix to only work with nearby stars
            self.active_signatures = self.signatures[self.lsh_candidates.active_indices, :]
            n_candidates = len(self.active_signatures)
            if n_candidates < n_stars:
                print "not all community members are active. Will only consider ", n_candidates, ' of the ', n_stars, ' stars'
                n_additions = n_candidates
            else:
                n_additions = n_stars
                # implement a new lookup
        else:
            self.active_signatures = self.signatures

        # find the jaccard distance to all non-seeds averaged over the seeds
        ast0 = time()
        star_similarities = self.calculate_initial_average_similarity(seeds)
        avg_sim_time = time() - ast0
        self.output_best_initial_averages(star_similarities, seeds, tags, community_folder, community_sizes,
                                          n_seeds, n_stars, result_interval, file_name='initial_avgs.csv')
        prt0 = time()
        R = self.pageRank(star_similarities, seeds, print_full_info=True)
        pr_time = time() - prt0
        self.output_best_initial_averages(R, seeds, tags, community_folder, community_sizes, n_seeds, n_stars,
                                          result_interval, file_name='pagerank.csv')

        self.used_ids = {}
        srt0 = time()
        for idx in range(n_additions):
            # Adds the next most similar star to each group of seeds and updates the average distance from the community members to all other stars
            self.increment_communities(star_similarities, seeds, excluded_set)
            # record the recall every
            if (idx + 1) % 10 == 0:
                print idx + 1, 'stars added'
        sim_rank_time = time() - srt0

        if runtime_file:
            writer = csv.writer(runtime_file)
            community = community_folder.rsplit('/', 1)[-1]
            writer.writerow(['page_rank', community, pr_time])
            writer.writerow(['min_rank', community, sim_rank_time])
            writer.writerow(['avg_sim_time', community, avg_sim_time])

        print 'added', n_stars, 'into each of', len(seeds), ' communities in ', time() - start_time, 'seconds'
        return seeds

    def load_lsh_table(self):
        """
        loads the locality sensitive hash lookup table into memory
        takes about 2 minutes
        """
        start = time()
        print 'reading pickled hash table'
        self.star_compression.load_lsh()
        self.lsh_table = self.star_compression.get_lsh()
        print 'table read into memory in time ', time() - start, ' s'


    def run_experimentation(self, n_seeds, group, random_seeds, result_interval, runtime_file)
        """
        performs a series of community detection runs using different seeds
        and measure recall
        :param n_seeds = 5 # The number of seeds to start with
        :param result_interval = 10 # the intervals in number of stars to snap the recall at
        """
        community_size = self.signatures['community'].value_counts()

        with open(self.outfolder + '/minrank.csv', 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_stars, result_interval)
            writer.writerow(cols)

        with open(self.outfolder + '/initial_avgs.csv', 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_stars, result_interval)
            writer.writerow(cols)

        with open(self.outfolder + '/pagerank.csv', 'wb') as f:
            writer = csv.writer(f)
            cols = xrange(result_interval, n_stars, result_interval)
            writer.writerow(cols)

            # load the tags file into a data frame
        tags = pd.io.parsers.read_csv(TAG_FILE_PATH)
        # add column for community affiliations
        tags.insert(0, 'community', -1)

        excluded_set = self.get_excluded_set()

        # Read the stars and associated community labels from file
        seeds = {}
        # The size of each ground truthed community - required to calculate recall
        community_sizes = {}

        start_time = time()

        for idx, rdm_seed in enumerate(random_seeds):
            seeds = self.signatures

            print seeds

            communities = self.run_community_detection(community_folder, tags, seeds, excluded_set, community_sizes,
                                                       n_stars=n_stars, n_seeds=n_seeds, country=country,
                                                       min_seed_followers=min_seed_followers, use_union=False,
                                                       max_followers=1e7,
                                                       generate_seeds=generate_seeds, result_interval=result_interval,
                                                       seed_file=None, runtime_file=runtime_file)
            print 'completed for ', n_stars, 'stars in ', time() - start_time

            # print 'community shape ',communities
            self.output_results(tags, communities, community_folder)
            self.calculate_recall(tags, communities, community_folder, community_sizes, n_seeds, n_stars,
                                  result_interval)

        print 'experimentation completed for ', len(random_seeds), ' random restarts in ', time() - start_time


if __name__ == '__main__':
    n_seeds = 30  # The number of seeds to start with
    result_interval = 10  # the intervals in number of stars to snap the recall at
    random_seeds = [451235, 35631241, 2315, 346213456, 134]

    inpath = 'local_resources/plos_one_data.csv'
    outfolder = 'results'

    data = pd.read_csv(inpath, index_col=0)
    data.index.name = 'community'
    data = data.reset_index()
    community_detector = CommunityDetector(data, outfolder, use_lsh=True)
    with open('results/runtimes_' + str(n_seeds) + '.csv', 'wb') as runtime_file:
        writer = csv.writer(runtime_file)
        writer.writerow(['community', 'method', 'runtime'])
    with open('local_resources/ICWSM15/runtimes_' + str(n_seeds) + '.csv', 'ab') as runtime_file:
        grouped = data.groupby('community')
        for group in grouped:
            community_detector.run_experimentation(n_seeds, group, random_seeds, result_interval, runtime_file)
