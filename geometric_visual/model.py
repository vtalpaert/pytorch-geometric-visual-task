import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.nn as gnn


class GeneralPurposeNet(nn.Module):
    def __init__(self, gconv1, gconv2, channels=3):
        super().__init__()
        self.global_features = nn.Sequential(
            nn.Conv2d(channels, 10, 11, stride=4, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(10, 16, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.gconv1 = gconv1
        self.gconv2 = gconv2

    def forward(self, data):
        image_features = self.global_features(data.im)
        repeated_image_features = torch.repeat_interleave(image_features, data.num_nodes_per_graph, dim=0)  # BN x 2
        x = torch.cat([data.x, repeated_image_features], dim=1)  # BN x 4
        graph_features = self.gconv2(F.leaky_relu_(self.gconv1(x, data.edge_index)), data.edge_index)
        out = graph_features.view(data.num_graphs, -1)
        return F.softmax(out, dim=1), image_features  # F.log_softmax(out, dim=1)


class Baseline(nn.Linear):
    def forward(self, x, edge_index):
        return super().forward(x)


def make_linear():
    lin1 = Baseline(4, 8)
    lin2 = Baseline(8, 1)
    return GeneralPurposeNet(lin1, lin2)

def make_gcnnet(aggrs=('max', 'max'), improved=(True, True)):
    def make_net():
        gconv1 = gnn.GCNConv(4, 8, bias=True, improved=improved[0], flow="source_to_target")
        gconv2 = gnn.GCNConv(8, 1, bias=True, improved=improved[1], flow="source_to_target")
        gconv1.aggr = aggrs[0]
        gconv2.aggr = aggrs[1]
        return GeneralPurposeNet(gconv1, gconv2)
    return make_net

def make_graphconvnet(aggrs=('max', 'max')):
    def make_net():
        gconv1 = gnn.GraphConv(4, 8, aggr=aggrs[0])
        gconv2 = gnn.GraphConv(8, 1, aggr=aggrs[1])
        return GeneralPurposeNet(gconv1, gconv2)
    return make_net

def make_SAGEconvnet(normalize=(False, False)):
    def make_net():
        gconv1 = gnn.SAGEConv(4, 8, normalize=normalize[0])
        gconv2 = gnn.SAGEConv(8, 1, normalize=normalize[1])
        return GeneralPurposeNet(gconv1, gconv2)
    return make_net

def make_GATconvnet(heads=1, concat=True):
    def make_net():
        gconv1 = gnn.GATConv(4, 8, heads=heads, concat=concat)
        gconv2 = gnn.GATConv(8, 1, heads=heads, concat=concat)
        return GeneralPurposeNet(gconv1, gconv2)
    return make_net

def make_TAGconvnet(K=3):
    def make_net():
        gconv1 = gnn.TAGConv(4, 8, K=K)
        gconv2 = gnn.TAGConv(8, 1, K=K)
        return GeneralPurposeNet(gconv1, gconv2)
    return make_net

def make_SGconvnet(K=1):
    def make_net():
        gconv1 = gnn.SGConv(4, 8, K=K)
        gconv2 = gnn.SGConv(8, 1, K=K)
        return GeneralPurposeNet(gconv1, gconv2)
    return make_net
