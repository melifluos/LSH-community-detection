# -*- coding: utf-8 -*-
"""
@author: ben chamberlain
"""
from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor
import numpy as np
from sklearn.metrics import auc
import csv  # to write out area under curves
from datetime import datetime
import os

from matplotlib import rcParams

rcParams['axes.labelsize'] = 6
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['legend.fontsize'] = 6
# rcParams['title.fontsize'] = 6

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
# rcParams['text.usetex'] = True

fig_dim = (3.1382316313823164, 1.9395338127643029)

METHODS = ['page_rank', 'initial_avgs', 'minrank']  # clustering methods to compare
COLOURS = ['r', 'b', 'y']  # colours to plot the different methods


def read_recall(folder, community, method):
    """
    read the recall files from disk
    :param folder: String. folder containing the data
    :param community: String. Name of the community. May contain spaces
    :param method: String. Name of the method
    :return:
    """
    try:
        filename = community.replace(" ", "_") + "_" + method + ".csv"
    except AttributeError:
        filename = str(community) + "_" + method + ".csv"
    data_path = os.path.join(folder, filename)
    df = pd.read_csv(data_path)
    rows, cols = df.shape
    index = df.columns.astype(np.int)
    index = index / max(index)
    mean = df.mean(axis=0)
    std_error = df.std(axis=0) / sqrt(rows)
    return index.values, mean.values, std_error.values


def uniform_sample(n, m):
    """
    n is the number to sample
    m is the length of the object being sampled
    """
    interval = m / n
    indices = [0]
    index = 0.0
    while True:
        index += interval
        if index >= m - 1:
            indices.append(int(m - 1))
            break
        else:
            indices.append(int(index))

    return np.array(indices)


def make_plot(folder, community, axarr=None, pos=None, n_points=20, show_legend=False, auc_file_writer=None):
    """
    folder - the location of the results
    fig,pos - pass a figure an coords in the figure to compound plot
    :param folder: the location of the results
    :param community: the name of the community
    :param axarr:
    :param pos:
    :param n_points:
    :param show_legend:
    :param auc_file_writer:
    :return:
    """
    if axarr == None:
        fig = plt.figure(figsize=fig_dim)
        ax = fig.add_subplot(pos[0], pos[1], 1, zorder=1)
    else:
        ax = axarr[pos[0], pos[1]]
    # ax.set_ylim(bottom=0)
    if pos[0] == len(axarr) - 1:
        ax.set_xlabel('additions as % of |C|', fontsize=10)
    # ax.set_xlabel(r'$\# \textup{inclusions}/ | C^{\textup{true}} |$', fontsize = 16)
    if pos[1] == 0:
        ax.set_ylabel('recall', fontsize=10)

    area_under_curve = []
    for idx, method in enumerate(METHODS):
        index, mean, std_error = read_recall(folder, community, method)
        area_under_curve.append(auc(index, mean))
        plot_indices = uniform_sample(n_points, len(
            index))  # don't want to plot all of the indices as the error bars look cluttered
        ax.errorbar(np.insert(index[plot_indices], 0, 0), np.insert(mean[plot_indices], 0, 0),
                    np.insert(std_error[plot_indices], 0, 0), color=COLOURS[idx], label=method,
                    alpha=0.5)
        # ax.set_title(folder.rsplit('/', 1)[-1], fontsize=10)
        ax.set_title(community, fontsize=8, y=0.95)
    auc_file_writer.writerow([community] + area_under_curve)
    if show_legend:
        legend = ax.legend(loc='upper center')
        for label in legend.get_texts():
            label.set_fontsize(6)

    if axarr == None:
        fig.tight_layout(pad=0.1)
        tag = folder.rsplit('/', 2)[-2]
        plt.savefig("local_results/recall_figs/recall_vs_range_{}.pdf".format(tag))
        plt.close()

    return index, mean, std_error


def get_communities_above_threshold(data, threshold_size):
    """
    return communities in data that occure more than threshold_size. Communities are encoded in the index
    :param data: A pandas dataframe with communities as the index
    :param threshold_size:
    :return:
    """
    index_counts = data.index.value_counts()
    count_list = index_counts[index_counts > threshold_size].index.values.tolist()
    return count_list


def plot_recall(data, community_folder, out_folder, threshold_size=0):
    """
    generate the experimental recall curves for pagerank, minrank and initial_avg
    :param data: A pandas dataframe indexed by community label
    :param community_folder: the location of the results
    :param out_folder: the location to write the graphs to
    :param threshold_size: the minimum number of members a community must have to plot
    :return:
    """
    data.index.name = 'community'
    communities = get_communities_above_threshold(data, threshold_size)
    n_communities = len(communities)
    print n_communities, ' communities containing more then ', threshold_size, ' members'
    # no longer used
    shape = int(ceil(sqrt(n_communities)))
    n_plot_rows = 5
    n_plot_cols = 3
    fig, axarr = plt.subplots(n_plot_rows, n_plot_cols, sharex='col', sharey='row')
    ypos = -1
    xpos = 0
    out_path = out_folder + '/area_under_curve' + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv'
    plot_path = out_folder + '/recall_vs_range' + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '.pdf'
    with open(out_path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['tags'] + METHODS)
    for idx, community in enumerate(communities):
        print community
        xpos = int(idx % n_plot_cols)
        if xpos == 0:
            ypos += 1
        with open(out_path, 'ab') as f:
            writer = csv.writer(f)
            make_plot(community_folder, community, axarr, (ypos, xpos), auc_file_writer=writer)

    for row in axarr:
        for ax in row:
            ax.set_ylim(bottom=0.)

    # fig.tight_layout(pad=0.1)
    plt.savefig(plot_path)
    plt.close('all')


if __name__ == '__main__':
    plt.close('all')
    inpath = '../../local_resources/email_data/signatures.txt'
    out_folder = '../../results/email'
    threshold = 25
    data = pd.read_csv(inpath, index_col=0)
    data.index.name = 'community'
    plot_recall(data, out_folder, out_folder, threshold)
