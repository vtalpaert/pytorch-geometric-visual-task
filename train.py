import torch_geometric.nn as gnn

from geometric_visual.tasks import FindNthClosestToCenterTask
from geometric_visual.model import make_linear, make_net, make_GINconvnet


models = {
    "baseline_dtch": make_linear(detach=True),  # good only for Task-0
    #"SAGE_10": make_net(gnn.SAGEConv, 4, 8, 1, [{'normalize': True}, {'normalize': False}]),
    #"SAGE_d3n1_dtch": make_net(gnn.SAGEConv, 4, 8, 1, 3 * [{'normalize': True}], detach=True),
    "Graph_d8_h4_mean_dtch": make_net(gnn.GraphConv, 4, 4, 1, 8 * [{'aggr': 'mean'}], detach=True),  # 67% accuracy, best perf of Graph variations
    #"GAT_d3h3c0_dtch": make_net(gnn.GATConv, 4, 8, 1, 3 * [{'heads': 3, 'concat': False}], detach=True),
    #"TAG_d2K7_h4_dtch_l": make_net(gnn.TAGConv, 4, 4, 1, 2 * [{'K': 7}], detach=True),  # fails
    #"TAG_d5K3_h4_dtch_l": make_net(gnn.TAGConv, 4, 4, 1, 5 * [{'K': 3}], detach=True),  # 53 % accuracy
    #"TAG_d5K3_h6_dtch": make_net(gnn.TAGConv, 4, 6, 1, 5 * [{'K': 3}], detach=True),  # unstable
    "TAG_d5K3_h4_dtch": make_net(gnn.TAGConv, 4, 4, 1, 5 * [{'K': 3}], detach=True),  # 67 % accuracy ?
    #"TAG_d7K0_h4_dtch_l": make_net(gnn.TAGConv, 4, 4, 1, 7 * [{'K': 0}], detach=True),  # 20% accuracy
    #"TAG_d7K1_h4_dtch_l": make_net(gnn.TAGConv, 4, 6, 1, 7 * [{'K': 1}], detach=True),  # 60% accuracy
    #"TAG_d3K012_dtch": make_net(gnn.TAGConv, 4, 6, 1, [{'K': 0}, {'K': 1}, {'K': 2}], detach=True),
    #"TAG_d3K123_dtch": make_net(gnn.TAGConv, 4, 6, 1, [{'K': 1}, {'K': 2}, {'K': 3}], detach=True),
    #"SG_d3_dtch_reverse": make_net(gnn.SGConv, 4, 6, 1, [{'K': 1}, {'K': 2}, {'K': 3}], detach=True),  # bad
    #"GIN5": make_GINconvnet(detach=True),
}


for position in range(3):
    task = FindNthClosestToCenterTask(position, True)
    task.benchmark(models)
