import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv


class GCNNet(nn.Module):
    def __init__(self, channels, flow="source_to_target"):
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
        self.gconv1 = GCNConv(4, 8, bias=True, improved=False, flow=flow)
        self.gconv2 = GCNConv(8, 1, bias=True, improved=True, flow=flow)
        self.gconv1.aggr = 'mean'
        self.gconv2.aggr = 'max'

    def forward(self, data):
        image_features = self.global_features(data.im)
        repeated_image_features = torch.repeat_interleave(image_features, data.num_nodes_per_graph, dim=0)  # BN x 2
        x = torch.cat([data.x, repeated_image_features], dim=1)  # BN x 4
        graph_features = self.gconv2(F.leaky_relu_(self.gconv1(x, data.edge_index)), data.edge_index)  # self.gconv2(F.leaky_relu_(self.gconv1(x, data.edge_index)), data.edge_index)
        out = graph_features.view(data.im.size(0), -1)
        return F.softmax(out, dim=1), image_features  # F.log_softmax(out, dim=1)
