import os
import pickle
import numpy as np
import scipy.sparse as sp
import time
from core.GraphDataBlock import GraphDataBlock
from util.graph_utils import neighbor_sampling
# from DGI.utils import process
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret


# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks


def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class EmbeddingDataSet():
    """
    Attributes:
        name (str): name of dataset
        data_dir (str): path to dataset folder
        train_dir (str): path to training data file
        test_dir (str): path to test data file
        input_dim (int): number of data features per node
        is_labelled (Boolean): whether underlying class labels are present
        all_data (list[GraphDataBlock]): data inputs packaged into blocks
        all_indices (np.array): input sequence when packaging data into blocks

        inputs (scipy csr matrix): data feature matrix of size n x f
        labels (np.array): data class label matrix of size n x 1
        adj_matrix (scipy csr matrix): adjacency matrix of size n x n
    """

    train_dir = {'cora': 'cora_full.pkl'}

    test_dir = train_dir

    def __init__(self, name, data_dir, train=True):
        self.name = name
        self.data_dir = data_dir

        if self.name == 'cora':
            self.train_dir = EmbeddingDataSet.train_dir[name]
            self.test_dir = EmbeddingDataSet.test_dir[name]
            self.is_labelled = False

            self.all_data = []

            # Extract data from file contents
            data_root = os.path.join(self.data_dir, self.name)
            if train:
                fname = os.path.join(data_root, self.train_dir)
            else:
                assert self.test_dir is not None
                fname = os.path.join(data_root, self.test_dir)
            with open(fname, 'rb') as f:
                file_contents = pickle.load(f)

            self.inputs = file_contents[0]
            self.labels = file_contents[1]
            self.adj_matrix = file_contents[2]

            self.is_labelled = len(self.labels) != 0
            self.input_dim = self.inputs.shape[1]

            self.all_indices = np.arange(0, self.inputs.shape[0])

            # Convert adj to csr matrix
            self.inputs = sp.csr_matrix(self.inputs)
            self.adj_matrix = sp.csr_matrix(self.adj_matrix)
            if train:
                print("train-data", self.inputs.shape)
            else:
                print("var-data", self.inputs.shape)

        else:
            adj, features, labels, idx_train, idx_val, idx_test = load_data(self.name)
            features, _ = preprocess_features(features)

            # print(adj.shape, features.shape)
            # A = adj
            # A = A.todense()

            nolabel = []
            for i, r in enumerate(labels):
                if all(r == 0):
                    nolabel.append(i)
            # print(nolabel)

            # L = labels
            L = []
            for i, r in enumerate(labels):
                if i in nolabel:
                    L.append(labels.shape[1])
                else:
                    L.append(np.where(r == 1)[0][0])

            # L = np.array([np.where(r == 1)[0][0] for r in labels])
            L = np.array(L)
            # print(L)
            self.is_labelled = False
            self.all_data = []

            # Extract data from file contents
            data_root = os.path.join(self.data_dir, self.name)
            if 1:
                # fname = os.path.join(data_root, self.train_dir)
                self.inputs = features
                self.labels = L
                self.adj_matrix = adj

                self.is_labelled = len(self.labels) != 0
                self.input_dim = self.inputs.shape[1]

                self.all_indices = np.arange(0, self.inputs.shape[0])

                # Convert adj to csr matrix
                self.inputs = sp.csr_matrix(self.inputs)
                self.adj_matrix = sp.csr_matrix(self.adj_matrix)
            # else:
            #     # assert self.test_dir is not None
            # #     # fname = os.path.join(data_root, self.test_dir)
            # # with open(fname, 'rb') as f:
            # #     file_contents = pickle.load(f)
            #     self.inputs = features[idx_val]
            #     self.labels = L[idx_val]
            #     self.adj_matrix = adj[idx_val]
            #
            #     self.is_labelled = len(self.labels) != 0
            #     self.input_dim = self.inputs.shape[1]
            #
            #     self.all_indices = np.arange(0, self.inputs.shape[0])
            #
            #     # Convert adj to csr matrix
            #     self.inputs = sp.csr_matrix(self.inputs)
            #     self.adj_matrix = sp.csr_matrix(self.adj_matrix)





    def create_all_data(self, n_batches=1, shuffle=False, sampling=False, full_path_matrix=None):
        """
        Initialises all_data as a list of GraphDataBlock
        Args:
            n_batches (int): number of blocks to return
            shuffle (Boolean): whether to shuffle input sequence
            sampling (Boolean): whether to expand data blocks with neighbor sampling
        """
        i = 0
        labels_subset = []
        self.all_data = []

        if shuffle:
            np.random.shuffle(self.all_indices)
        else:
            self.all_indices = np.arange(0, self.inputs.shape[0])

        # Split equally
        # TODO: Another option to split randomly
        chunk_sizes = self.get_k_equal_chunks(self.inputs.shape[0], k=n_batches)

        t_start = time.time()

        for num_samples in chunk_sizes:
            mask = sorted(self.all_indices[i: i + num_samples])

            # Perform sampling to obtain local neighborhood of mini-batch
            if sampling:
                D_layers = [9, 14]  # max samples per layer
                mask = neighbor_sampling(self.adj_matrix, mask, D_layers)

            inputs_subset = self.inputs[mask]
            adj_subset = self.adj_matrix[mask, :][:, mask]

            if self.is_labelled:
                labels_subset = self.labels[mask]

            # Package data into graph block
            G = GraphDataBlock(inputs_subset, labels=labels_subset, W=adj_subset)

            # Add original indices from the complete dataset
            G.original_indices = mask

            # Add shortest path matrix from precomputed data if needed
            if full_path_matrix is not None:
                G.precomputed_path_matrix = full_path_matrix[mask, :][:, mask]

            self.all_data.append(G)
            i += num_samples

        t_elapsed = time.time() - t_start
        print('Data blocks of length: ', [len(G.labels) for G in self.all_data])
        print("Time to create all data (s) = {:.4f}".format(t_elapsed))

    def summarise(self):
        print("Name of dataset = {}".format(self.name))
        print("Input dimension = {}".format(self.input_dim))
        print("Number of training samples = {}".format(self.inputs.shape[0]))
        print("Training labels = {}".format(self.is_labelled))

    def get_k_equal_chunks(self, n, k):
        # returns n % k sub-arrays of size n//k + 1 and the rest of size n//k
        p, r = divmod(n, k)
        return [p + 1 for _ in range(r)] + [p for _ in range(k - r)]

    def get_current_inputs(self):
        inputs = self.inputs[self.all_indices]
        labels = self.labels[self.all_indices]
        adj = self.adj_matrix[self.all_indices, :][:, self.all_indices]
        return inputs, labels, adj

    def get_sample_block(self, n_initial, sample_neighbors, verbose=0):
        """
        Returns a subset of data as a GraphDataBlock
        Args:
            n_initial (int): number of samples at the start
            sample_neighbors (Boolean): whether to expand the sample block with neighbor sampling
        Returns:
            G (GraphDataBlock): data subset
        """

        mask = sorted(np.random.choice(self.all_indices, size=n_initial, replace=False))
        if sample_neighbors:
            mask = neighbor_sampling(self.adj_matrix, mask, D_layers=[9, 14])
        inputs = self.inputs[mask]
        labels = self.labels[mask]
        W = self.adj_matrix[mask, :][:, mask]
        G = GraphDataBlock(inputs, labels, W)
        G.original_indices = mask
        if verbose:
            print("Initial set of {} points was expanded to {} points".format(n_initial, len(mask)))
        return G
