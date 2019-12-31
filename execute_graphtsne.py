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
from util.training_utils import save_metadata, save_train_log
from util.plot_graph_embedding import plot_graph_embedding
from util.network_utils import get_net_projection
import datetime

input_dir = 'data'
output_dir = 'results'
# dataset_name = 'cora'
# dataset_name = 'citeseer'
dataset_name = 'pubmed'

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

task_parameters = {}
task_parameters['n_components'] = 2
task_parameters['val_flag'] = True

net_parameters = {}
net_parameters['n_components'] = task_parameters['n_components']
net_parameters['D'] = dataset.input_dim  # input dimension
net_parameters['H'] = 128  # number of hidden units
net_parameters['L'] = 2  # number of hidden layers

iter_n = 10
emb_lst = []
time_lst = []
for i in range(iter_n):
    # Initialise network
    net = GraphConvNet(net_parameters)
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

    now = datetime.datetime.now()
    now_str = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)
    file_path = 'embed_' + "GraphTSNE_" + dataset_name + now_str
    # file_path = 'cora_data_graphtsne'
    # file_path2 = 'embed_' + dataset + '_' + "DGI-UMAP_" + str(iter_number+1) + 'th'

    # print("X:", dataset.inputs.todense())
    flag = 0
    if flag == 0:
        np.savez(file_path, X = dataset.inputs.todense(), L = dataset.labels, emb = emb_lst, A=dataset.adj_matrix.todense())
    else:
        np.savez(file_path, emb = emb_lst)
    flag = flag + 1

    # np.savez(file_path2, X = org_features, L = L, emb = emb_DGIUMAP_lst, A=A)