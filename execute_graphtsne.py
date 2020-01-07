import matplotlib.pyplot as plt

import os
import torch
import argparse
import pathlib
import time
import numpy as np

from learn_embedding import train
from core.EmbeddingDataSet import EmbeddingDataSet
from core.GraphConvNet import GraphConvNet
from core.GraphConvNet_DGI import GraphConvNetDGI
from util.training_utils import save_metadata, save_train_log
from util.plot_graph_embedding import plot_graph_embedding
from util.network_utils import get_net_projection
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from munkres import Munkres
import csv
from numpy import savetxt
from pandas import DataFrame
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, f1_score
import os
import glob
from matplotlib.backends.backend_pdf import PdfPages
# from MantelTest import Mantel
from hub_toolbox.distances import euclidean_distance
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numba
# From: https://github.com/leowyy/GraphTSNE/blob/master/util/evaluation_metrics.py

import numpy as np
from sklearn import neighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import shortest_path
import time


def kmeans_acc_ari_ami_f1(X, L, verbose=1):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    n_clusters = len(np.unique(L))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)

    y_pred = kmeans.fit_predict(X)
    y_pred = y_pred.astype(np.int64)
    y_true = L.astype(np.int64)
    assert y_pred.size == y_true.size

    y_pred = y_pred.reshape((1, -1))
    y_true = y_true.reshape((1, -1))

    # D = max(y_pred.max(), L.max()) + 1
    # w = np.zeros((D, D), dtype=np.int64)
    # for i in range(y_pred.size):
    #     w[y_pred[i], L[i]] += 1
    # # from sklearn.utils.linear_assignment_ import linear_assignment
    # from scipy.optimize import linear_sum_assignment
    # row_ind, col_ind = linear_sum_assignment(w.max() - w)
    #
    # return sum([w[i, j] for i in row_ind for j in col_ind]) * 1.0 / y_pred.size

    if len(np.unique(y_pred)) == len(np.unique(y_true)):
        C = len(np.unique(y_true))

        cost_m = np.zeros((C, C), dtype=float)
        for i in np.arange(0, C):
            a = np.where(y_pred == i)
            # print(a.shape)
            a = a[1]
            l = len(a)
            for j in np.arange(0, C):
                yj = np.ones((1, l)).reshape(1, l)
                yj = j * yj
                cost_m[i, j] = np.count_nonzero(yj - y_true[0, a])

        mk = Munkres()
        best_map = mk.compute(cost_m)

        (_, h) = y_pred.shape
        for i in np.arange(0, h):
            c = y_pred[0, i]
            v = best_map[c]
            v = v[1]
            y_pred[0, i] = v

        acc = 1 - (np.count_nonzero(y_pred - y_true) / h)

    else:
        acc = 0
    # print(y_pred.shape)
    y_pred = y_pred[0]
    y_true = y_true[0]
    ari, ami = adjusted_rand_score(y_true, y_pred), adjusted_mutual_info_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    results = {}
    results["ACC"] = acc
    results["ARI"] = ari
    results['AMI'] = ami
    results['F1_score'] = f1
    if verbose:
        for k, v in results.items():
            print("{} = {:.4f}".format(k, v))

    return acc, ari, ami, f1

input_dir = 'data'
output_dir = 'results'
dataset_name = 'cora'
# dataset_name = 'citeseer'
# dataset_name = 'pubmed'

# Here we set the graph clustering weight (alpha) at 0.4, as in the paper
opt_parameters = {'graph_weight': 0.4}
opt_parameters['learning_rate'] = 0.00075  # ADAM
opt_parameters['max_iters'] = 240
opt_parameters['batch_iters'] = 40
opt_parameters['save_flag'] = True
opt_parameters['decay_rate'] = 1.25
opt_parameters['start_epoch'] = 0

opt_parameters['distance_metric'] = 'cosine'
opt_parameters['n_batches'] = 1
opt_parameters['shuffle_flag'] = False
opt_parameters['sampling_flag'] = False
opt_parameters['val_batches'] = 1
opt_parameters['perplexity'] = 30

dataset = EmbeddingDataSet(dataset_name, input_dir, train=True)
dataset.summarise()
print("Feature", dataset.inputs.shape)

task_parameters = {}
task_parameters['n_components'] = 2
task_parameters['val_flag'] = True

net_parameters = {}
net_parameters['n_components'] = task_parameters['n_components']
net_parameters['D'] = dataset.input_dim  # input dimension
net_parameters['H'] = 256  # number of hidden units
net_parameters['L'] = 2  # number of hidden layers

iter_n = 1
emb_lst = []
time_lst = []
for i in range(iter_n):
    # Initialise network
    # net = GraphConvNet(net_parameters)
    net = GraphConvNetDGI(net_parameters)
    if torch.cuda.is_available():
        net.cuda()

    # Create checkpoint dir
    subdirs = [x[0] for x in os.walk(output_dir) if dataset_name in x[0]]
    run_number = str(len(subdirs) + 1)
    checkpoint_dir = os.path.join(output_dir, dataset_name + '_' + run_number)
    pathlib.Path(checkpoint_dir).mkdir(exist_ok=True)  # create the directory if it doesn't exist

    print("Number of network parameters = {}".format(net.nb_param))
    print('Saving results into: {}'.format(checkpoint_dir))

    # Start training here
    t_start = time.time()
    val_dataset = None
    if task_parameters['val_flag']:
        val_dataset = EmbeddingDataSet(dataset_name, input_dir, train=False)

    tab_results = train(net, dataset, opt_parameters, checkpoint_dir, val_dataset)

    end_epoch = opt_parameters['start_epoch'] + opt_parameters['max_iters']

    if opt_parameters['save_flag']:
        save_metadata(checkpoint_dir, task_parameters, net_parameters, opt_parameters, end_epoch)
        save_train_log(checkpoint_dir, tab_results, end_epoch)

    t_elapsed = time.time() - t_start
    print("Time elapsed = {:.4f}".format(t_elapsed))

    time_lst.append(t_elapsed)

    y_pred = get_net_projection(net, dataset)
    emb_lst.append(y_pred)
    plot_graph_embedding(y_pred, dataset.labels, dataset.adj_matrix, line_alpha=0.05, s=1)
    plt.show()

    print('elapsed time:', time_lst)

    kmeans_acc_ari_ami_f1(y_pred, dataset.labels)

    now = datetime.datetime.now()
    now_str = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)
    # file_path = 'embed_' + "GraphTSNE_" + dataset_name + now_str
    # file_path = 'cora_data_graphtsne'
    # file_path2 = 'embed_' + dataset + '_' + "DGI-UMAP_" + str(iter_number+1) + 'th'

    # print("X:", dataset.inputs.todense())
    # flag = 0
    # if flag == 0:
    #     np.savez(file_path, X = dataset.inputs.todense(), L = dataset.labels, emb = emb_lst, A=dataset.adj_matrix.todense())
    # else:
    #     np.savez(file_path, emb = emb_lst)
    # flag = flag + 1

    # np.savez(file_path2, X = org_features, L = L, emb = emb_DGIUMAP_lst, A=A)