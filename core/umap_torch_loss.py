import numpy as np
import torch
from time import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold.t_sne import _joint_probabilities, _joint_probabilities_nn
from scipy.spatial.distance import squareform

from util.graph_utils import get_shortest_path_matrix
from util.training_utils import get_torch_dtype
from scipy import optimize

dtypeFloat, dtypeLong = get_torch_dtype()

def prob_high_dim(D, rho, sigma, dist_row):
    """
    For each row of Euclidean distance matrix (dist_row) compute
    probability in high dimensions (1D array)
    """
    d = D[dist_row] - rho[dist_row]
    d[d < 0] = 0
    return np.exp(- d / sigma)



def sigma_binary_search(k_of_sigma, fixed_k):
    """
    Solve equation k_of_sigma(sigma) = fixed_k
    with respect to sigma by the binary search algorithm
    """
    sigma_lower_limit = 0; sigma_upper_limit = 1000
    for i in range(20):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        if k_of_sigma(approx_sigma) < fixed_k:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
            break
    return approx_sigma

def calculate_ab(min_dist):
    MIN_DIST = min_dist

    x = np.linspace(0, 3, 300)

    def f(x, min_dist):
        y = []
        for i in range(len(x)):
            if (x[i] <= min_dist):
                y.append(1)
            else:
                y.append(np.exp(- x[i] + min_dist))
        return y

    dist_low_dim = lambda x, a, b: 1 / (1 + a * x ** (2 * b))

    p, _ = optimize.curve_fit(dist_low_dim, x, f(x, MIN_DIST))

    a = p[0]
    b = p[1]
    print("Hyperparameters a = " + str(a) + " and b = " + str(b))
    return a, b

def prob_low_dim(Y, min_dist=0.1):
    """
    Compute matrix of probabilities q_ij in low-dimensional space
    """
    a, b = calculate_ab(min_dist)
    inv_distances = np.power(1 + a * pairwise_distances(Y, metric='euclidean', squared=True)**b, -1)

    return inv_distances

def compute_joint_probabilities(X, perplexity=30, n_neighbor=15, metric='euclidean', method='exact', adj=None, verbose=0):
    """
    Computes the joint probability matrix P from a feature matrix X of size n x f
    Adapted from sklearn.manifold.t_sne
    """

    def k(prob):
        """
        Compute n_neighbor = k (scalar) for each 1D array of high-dimensional probability
        """
        return np.power(2, np.sum(prob))

    # Compute pairwise distances
    if verbose > 0: print('Computing pairwise distances...')

    if method == 'exact':
        if metric == 'precomputed':
            D = X
        elif metric == 'euclidean':
            D = pairwise_distances(X, metric=metric, squared=True)
        elif metric == 'cosine':
            D = pairwise_distances(X, metric=metric)
        elif metric == 'shortest_path':
            assert adj is not None
            D = get_shortest_path_matrix(adj, verbose=verbose)

        # P = _joint_probabilities(D, desired_perplexity=perplexity, verbose=verbose)
        n = X.shape[0]
        rho = [sorted(D[i])[1] for i in range(D.shape[0])]
        prob = np.zeros((n, n))
        sigma_array = []
        for dist_row in range(n):
            func = lambda sigma: k(prob_high_dim(D, rho, sigma, dist_row))
            binary_search_result = sigma_binary_search(func, n_neighbor)
            prob[dist_row] = prob_high_dim(D, rho, binary_search_result, dist_row)
            sigma_array.append(binary_search_result)
            if (dist_row + 1) % 100 == 0:
                print("Sigma binary search finished {0} of {1} cells".format(dist_row + 1, n))
        print("\nMean sigma = " + str(np.mean(sigma_array)))

        P = prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))
        # P = (prob + np.transpose(prob)) / 2

        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be non-negative"
        assert np.all(P <= 1), ("All probabilities should be less "
                                "or then equal to one")

        # P = squareform(P)

    else:
        # Cpmpute the number of nearest neighbors to find.
        # LvdM uses 3 * perplexity as the number of neighbors.
        # In the event that we have very small # of points
        # set the neighbors to n - 1.
        n_samples = X.shape[0]
        k = min(n_samples - 1, int(3. * perplexity + 1))

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='auto', n_neighbors=k,
                               metric=metric)
        t0 = time()
        knn.fit(X)
        duration = time() - t0
        if verbose:
            print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                n_samples, duration))

        t0 = time()
        distances_nn, neighbors_nn = knn.kneighbors(
            None, n_neighbors=k)
        duration = time() - t0
        if verbose:
            print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
                  .format(n_samples, duration))

        # Free the memory used by the ball_tree
        del knn

        if metric == "euclidean":
            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            distances_nn **= 2

        # compute the joint probability distribution for the input space
        P = _joint_probabilities_nn(distances_nn, neighbors_nn,
                                    perplexity, verbose)
        P = P.toarray()

    # Convert to torch tensor
    P = torch.from_numpy(P).type(dtypeFloat)
    n = X.shape[0]
    P *= 1 - torch.eye(n).type(dtypeFloat)
    # P /= torch.sum(P)
    P /= torch.reshape(torch.sum(P, dim=1), [-1, 1])

    return P


def tsne_torch_loss(P, y_emb):
    """
    Computes the t-SNE loss, i.e. KL(P||Q). Torch implementation allows from auto-grad.
    Args:
        P (np.array): joint probabilities matrix of size n x n
        y_emb (np.array): low dimensional map of data points, matrix of size n x 2
    Returns:
        C (float): t-SNE loss
    """

    d = 2
    n = P.shape[1]
    v = d - 1.  # degrees of freedom
    eps = 10e-15  # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)

    # Euclidean pairwise distances in the low-dimensional map
    sum_act = torch.sum(y_emb.pow(2), dim=1)
    Q = sum_act + torch.reshape(sum_act, [-1, 1]) + -2 * torch.mm(y_emb, torch.t(y_emb))

    Q = Q / v
    Q = torch.pow(1 + Q, -(v + 1) / 2)
    Q *= 1 - torch.eye(n).type(dtypeFloat)
    Q /= torch.sum(Q)
    Q = torch.clamp(Q, min=eps)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C

def umap_torch_loss(P, y_emb, a, b):
    # print(P)

    eps = 10e-8
    n = P.shape[0]
    # a = torch.from_numpy(a).type(dtypeFloat)
    # b = torch.from_numpy(b).type(dtypeFloat)
    Q = torch.norm(y_emb[:, None] - y_emb, dim=2, p=2)
    # print("Q", Q.shape, Q.dtype)
    # sum_act = torch.sum(y_emb.pow(2), dim=1)
    # Q = sum_act + torch.reshape(sum_act, [-1, 1]) + -2 * torch.mm(y_emb, torch.t(y_emb))
    # print("Q's shape:", Q.shape)
    Q = torch.pow(Q, 2 * b)
    Q = 1 + a * Q
    Q = torch.pow(Q, -1)
    # print(P[:, 0])
    # Q = prob_low_dim(y_emb, min_dist)
    # Q = torch.from_numpy(Q).type(dtypeFloat)
    Q *= 1 - torch.eye(n).type(dtypeFloat)
    # Q /= torch.sum(Q)
    Q /= torch.reshape(torch.sum(Q, dim=1), [-1, 1])
    Q = torch.clamp(Q, min=eps)

    CE = - P * torch.log(Q + eps) - (1 - P) * torch.log(1 - Q + eps)
    CE = torch.sum(CE)

    # CE = torch.log((P + eps) / (Q + eps))
    # CE = torch.sum(P * CE)
    return CE