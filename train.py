from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from geometric_visual.dataset import GeoVisualDataset
from geometric_visual.model import *
from geometric_visual.training import benchmark


IM_CHANNELS = 3
IM_HEIGHT = IM_WIDTH = 64
NUM_NODES = 10
RADIUS = 0.3
MAX_NUM_NEIGHBORS = 3
TARGET_COLOR = (1.,0.,0.)
BATCH_SIZE = 64
AUXILIARY_TASK_WEIGHT = 3
NUM_WORKERS = 8  # cpu workers to generate data, use 0 for GPU dataset (not recommended)
SIZE_TRAIN = 8000
SIZE_TEST = 2000
MAX_EPOCHS = 200


train_dataset = GeoVisualDataset(SIZE_TRAIN, NUM_NODES, IM_HEIGHT, IM_WIDTH, TARGET_COLOR, radius=RADIUS, max_num_neighbors=MAX_NUM_NEIGHBORS, device='cpu')
test_dataset = GeoVisualDataset(SIZE_TEST, NUM_NODES, IM_HEIGHT, IM_WIDTH, TARGET_COLOR, radius=RADIUS, max_num_neighbors=MAX_NUM_NEIGHBORS, device='cpu')
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False)

models = {
    "baseline_lin": make_linear,
    #"GCN_max1_max1": make_gcnnet(aggrs=('max', 'max'), improved=(True, True)),
    #"GCN_mean0_mean0": make_gcnnet(aggrs=('mean', 'mean'), improved=(False, False)),
    #"GCN_max1_mean0": make_gcnnet(aggrs=('max', 'mean'), improved=(True, False)),
    #"GCN_mean1_max1": make_gcnnet(aggrs=('mean', 'max'), improved=(True, True)),
    #"GCN_add0_add0": make_gcnnet(aggrs=('add', 'add'), improved=(False, False)),
    #"SAGE_10": make_SAGEconvnet(normalize=(True, False)),
    #"SAGE_00": make_SAGEconvnet(normalize=(False, False)),
    ##"Graph_max_max": make_graphconvnet(aggrs=('max', 'max')),  # bad perf
    "Graph_mean_mean": make_graphconvnet(aggrs=('mean', 'mean')),  # best perf
    #"Graph_mean_max": make_graphconvnet(aggrs=('mean', 'max')),  # bad
    #"Graph_add_add": make_graphconvnet(aggrs=('add', 'add')),
    #"GAT_h1c1": make_GATconvnet(heads=1, concat=True),
    ##"GAT_h1c0": make_GATconvnet(heads=1, concat=False),
    "TAG_K1": make_TAGconvnet(1),
    #"TAG_K3": make_TAGconvnet(3),
    "TAG_K9": make_TAGconvnet(9),
    #"SG_K1": make_SGconvnet(1),
    #"SG_K3": make_SGconvnet(3),
}

for experiment_name, model_fun in models.items():
    experiment_name += '_' + str(NUM_NODES).zfill(3)
    model = model_fun()
    benchmark(model, train_loader, test_loader, auxiliary_task_weight=AUXILIARY_TASK_WEIGHT, comment=experiment_name, max_epochs=MAX_EPOCHS)
