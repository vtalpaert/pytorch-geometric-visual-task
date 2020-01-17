from geometric_visual.tasks.closest_to_center import FindClosestToCenterTask
from geometric_visual.model import *

models = {
    "baseline": make_linear,
    #"GCN_max1_max1": make_gcnnet(aggrs=('max', 'max'), improved=(True, True)),
    #"GCN_mean0_mean0": make_gcnnet(aggrs=('mean', 'mean'), improved=(False, False)),
    #"GCN_max1_mean0": make_gcnnet(aggrs=('max', 'mean'), improved=(True, False)),
    #"GCN_mean1_max1": make_gcnnet(aggrs=('mean', 'max'), improved=(True, True)),
    #"GCN_add0_add0": make_gcnnet(aggrs=('add', 'add'), improved=(False, False)),
    #"SAGE_10": make_SAGEconvnet(normalize=(True, False)),
    #"SAGE_00": make_SAGEconvnet(normalize=(False, False)),
    ##"Graph_max_max": make_graphconvnet(aggrs=('max', 'max')),  # bad perf
    #"Graph_mean_mean": make_graphconvnet(aggrs=('mean', 'mean')),  # best perf
    #"Graph_mean_max": make_graphconvnet(aggrs=('mean', 'max')),  # bad
    #"Graph_add_add": make_graphconvnet(aggrs=('add', 'add')),
    #"GAT_h1c1": make_GATconvnet(heads=1, concat=True),
    ##"GAT_h1c0": make_GATconvnet(heads=1, concat=False),
    #"TAG_K1": make_TAGconvnet(1),
    #"TAG_K3": make_TAGconvnet(3),
    #"TAG_K9": make_TAGconvnet(9),
    #"SG_K1": make_SGconvnet(1),
    #"SG_K3": make_SGconvnet(3),
}

task = FindClosestToCenterTask()
task.benchmark(models)
