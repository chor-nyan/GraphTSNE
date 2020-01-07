import torch
import torch.nn as nn
from core.GraphConvNet import GraphConvNet
from layers import AvgReadout, Discriminator, PermutateGraph
import numpy as np

class GraphConvNetDGI(nn.Module):
    def __init__(self, net_parameters):
        super(GraphConvNetDGI, self).__init__()

        self.name = 'graph_net_DGI'

        self.gcn = GraphConvNet(net_parameters)

        # parameters
        D = net_parameters['D']
        n_components = net_parameters['n_components']
        H = net_parameters['H']
        L = net_parameters['L']

        self.D = D

        self.fc = nn.Linear(H, 2)
        # init
        self.init_weights_Graph_OurConvNet(H, 2, 1)

        self.disc = Discriminator(H)


        # list = []

    def forward(self, G):
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.perm = PermutateGraph()

        shuffled_features = self.perm(G)
        h_2 = self.gcn(G, feature=shuffled_features)

        h_1 = self.gcn(G)
        x = self.fc(h_1)

        c = self.read(h_1)
        c = self.sigm(c)

        # h_1, h_2 = h_1.data.numpy(), h_2.data.numpy()
        # print("h_1:", h_1.shape, "h_2:", h_2.shape, "c:", c.shape)
        # print("h_1 and h_2 are same?", h_1[0], h_2[0])
        ret = self.disc(c, h_1, h_2)

        # ret = 1


        return x, ret

    # # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = self.gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()

    def init_weights_Graph_OurConvNet(self, Fin_fc, Fout_fc, gain):

        scale = gain * np.sqrt(2.0 / Fin_fc)
        self.fc.weight.data.uniform_(-scale, scale)
        self.fc.bias.data.fill_(0)


    def loss(self, y, y_target):
        loss = nn.MSELoss()(y, y_target) # L2 loss
        return loss

    def pairwise_loss(self, y, y_target, W):
        distances_1 = y_target[W.row, :] - y_target[W.col, :]
        distances_2 = y[W.row, :] - y[W.col, :]
        loss = torch.mean(torch.pow(distances_1.norm(dim=1) - distances_2.norm(dim=1), 2))

        return loss

    def update(self, lr):
        update = torch.optim.Adam(self.parameters(), lr=lr)
        return update

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    def nb_param(self):
        return self.D