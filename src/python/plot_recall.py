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
# METHODS = ['pagerank', 'initial_avgs']  # clustering methods to compare
COLOURS = ['r', 'b', 'y']  # colours to plot the different methods


def read_recall(folder, community, method):
    """
    read the recall files from disk
    :param folder: String. folder containing the data
    :param community: String. Name of the community. May contain spaces
    :param method: String. Name of the method
    :return:
    """
    filename = community.replace(" ", "_") + "_" + method + ".csv"
    data_path = os.path.join(folder, filename)
    df = pd.read_csv(data_path)
    rows, cols = df.shape
    index = df.columns.astype(np.int)
    index = index / max(index)
    mean = df.iloc[range(1, rows, 2), :].mean()
    std_error = df.iloc[range(1, rows, 2), :].std() / sqrt(rows)
    return index, mean, std_error


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
        ax.errorbar(index[plot_indices], mean[plot_indices], std_error[plot_indices], color=COLOURS[idx], label=method,
                    alpha=0.5)
        ax.set_title(folder.rsplit('/', 1)[-1], fontsize=10)
    writer.writerow([community] + area_under_curve)
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


if __name__ == '__main__':
    plt.close('all')
    inpath = '../../local_resources/twitter_data.csv'
    outfolder = '../../results'

    data = pd.read_csv(inpath, index_col=0)
    data.index.name = 'community'

    communities = data.index.unique().values
    community_folder = '../results'  # Where to get the data for the plots
    n_communities = len(communities)
    # no longer used
    shape = int(ceil(sqrt(n_communities)))
    n_plot_rows = 4
    n_plot_cols = 4
    fig, axarr = plt.subplots(n_plot_rows, n_plot_cols, sharex='col', sharey='row')
    ypos = -1
    xpos = 0
    with open('../../local_results/area_under_curve' + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['tags'] + METHODS)
    for idx, community in enumerate(communities):
        print community
        xpos = int(idx % n_plot_cols)
        if xpos == 0:
            ypos += 1
        with open('../../local_results/area_under_curve' + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv', 'ab') as f:
            writer = csv.writer(f)
            make_plot(outfolder, community, axarr, (ypos, xpos), auc_file_writer=writer)

    for row in axarr:
        for ax in row:
            ax.set_ylim(bottom=0.)

    # fig.tight_layout(pad=0.1)
    plt.savefig("../../local_results/recall_figs/recall_vs_range" + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".pdf")
    plt.close('all')
