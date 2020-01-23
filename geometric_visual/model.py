import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.nn as gnn


class GraphSequential(nn.Sequential):
    """Sequential module which repeats the initial edge_index to all submodules"""
    def forward(self, x, edge_index):
        for module in self._modules.values():
            x = module(x, edge_index)
        return x


def Graph(Module):
    """Makes a "classic" module compatible with torch_geometric"""
    class GraphModule(Module):
        def forward(self, x, edge_index):
            return super().forward(x)
    return GraphModule


class CustomTanh(nn.Tanh):
    """Replaces the Sigmoid for stability"""
    def forward(self, input):
        return torch.tanh(input * 2 / 3) * 0.858 + 0.5


class GeneralPurposeNet(nn.Module):
    """Extracts image features, concat to node features and pass to gconv"""
    def __init__(self, gconv, channels=3, detach=False):
        # keep in mind that detach=True will clearly simplify the task
        super().__init__()
        self.detach = detach
        self.global_features = nn.Sequential(
            nn.Conv2d(channels, 16, 11, stride=4, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(16, 16, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 2),
            CustomTanh(),  # nn.Sigmoid()
        )
        self.gconv = gconv

    def forward(self, data):
        image_features = self.global_features(data.im)
        _image_features = image_features.detach() if self.detach else image_features
        #_image_features.requires_grad = self.training
        repeated_image_features = torch.repeat_interleave(_image_features, data.num_nodes_per_graph, dim=0)  # BN x 2
        x = torch.cat([data.x, repeated_image_features], dim=1)  # BN x 4
        graph_features = self.gconv(x, data.edge_index)
        out = graph_features.view(data.num_graphs, -1)
        return F.softmax(out, dim=1), image_features  # F.log_softmax(out, dim=1)


def make_linear(detach):
    """Baseline network. Two layers linear net"""
    def fun():
        return GeneralPurposeNet(GraphSequential(
            Graph(nn.Linear)(4, 8),
            Graph(nn.LeakyReLU)(inplace=True),
            Graph(nn.Linear)(8, 1),
        ), detach=detach)
    return fun


def make_net(cls, in_channels, hidden_channels, out_channels, kwargs_list, detach):
    """Returns function for network creation, repeats the module of class cls in a GraphSequential"""
    def fun():
        seq = GraphSequential()
        idx = 0
        for cls_idx, kwargs in enumerate(kwargs_list):
            ic = in_channels if cls_idx == 0 else hidden_channels
            oc = out_channels if cls_idx == len(kwargs_list) - 1 else hidden_channels
            seq.add_module(str(idx), cls(ic, oc, **kwargs))
            idx += 1
            if idx != len(kwargs_list) - 1:
                seq.add_module(str(idx), Graph(nn.LeakyReLU)(inplace=True))
                idx += 1
        return GeneralPurposeNet(seq, detach=detach)
    return fun


def make_GINconvnet(detach):
    def fun():
        return GeneralPurposeNet(GraphSequential(
            gnn.GINConv(nn.Linear(4, 6), eps=1e-6, train_eps=True),
            Graph(nn.LeakyReLU)(inplace=True),
            gnn.GINConv(nn.Linear(6, 6), eps=1e-6, train_eps=True),
            Graph(nn.LeakyReLU)(inplace=True),
            gnn.GINConv(nn.Linear(6, 6), eps=1e-6, train_eps=True),
            Graph(nn.LeakyReLU)(inplace=True),
            gnn.GINConv(nn.Linear(6, 6), eps=1e-6, train_eps=True),
            Graph(nn.LeakyReLU)(inplace=True),
            gnn.GINConv(nn.Linear(6, 1), eps=1e-6, train_eps=True),
        ), detach=detach)
    return fun
