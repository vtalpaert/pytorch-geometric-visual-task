from time import process_time
from math import log

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from geometric_visual.dataset import GeoVisualDataset
from geometric_visual.draw import draw_n

class FindClosestToCenterTask(object):
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
    EPOCHS = 200

    def __init__(self):
        train_dataset = GeoVisualDataset(
            self.SIZE_TRAIN,
            self.NUM_NODES,
            self.IM_HEIGHT,
            self.IM_WIDTH,
            self.TARGET_COLOR,
            radius=self.RADIUS,
            max_num_neighbors=self.MAX_NUM_NEIGHBORS,
            device='cpu'
        )
        test_dataset = GeoVisualDataset(
            self.SIZE_TEST,
            self.NUM_NODES,
            self.IM_HEIGHT,
            self.IM_WIDTH,
            self.TARGET_COLOR,
            radius=self.RADIUS,
            max_num_neighbors=self.MAX_NUM_NEIGHBORS,
            device='cpu'
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS, drop_last=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_once(self, model, optimizer):
        model.train()
        start = process_time()
        losses = {
            "probs": 0,
            "aux": 0,
            "total": 0
        }
        N = self.train_loader.dataset.num_nodes
        loss_probs_norm = (log(N)-(N-1)*log(1-1/N))/N
        for data in self.train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            probs, aux = model(data)
            target = data.y.view(data.num_graphs, -1).type(dtype=torch.float)
            num_nodes = target.size(1)
            loss_probs = F.binary_cross_entropy(probs, target, reduction='mean') #F.cross_entropy(scores, data.nclose)  # F.binary_cross_entropy_with_logits
            loss_probs /= loss_probs_norm
            loss_aux = F.mse_loss(aux, data.square_center, reduction='mean')
            loss_aux *= 4.5
            loss =  (loss_probs + self.AUXILIARY_TASK_WEIGHT * loss_aux) / (1 + self.AUXILIARY_TASK_WEIGHT)
            loss.backward()
            losses["total"] += data.num_graphs * loss.item()
            losses["probs"] += data.num_graphs * loss_probs.item()
            losses["aux"] += data.num_graphs * loss_aux.item()
            optimizer.step()
        end = process_time()
        size = self.train_loader.dataset.size
        losses["total"] /= size
        losses["probs"] /= size
        losses["aux"] /= size
        return losses, end - start
    
    def test(self, model):
        model.eval()
        correct = 0
        for i, data in enumerate(self.test_loader):
            with torch.no_grad():
                data = data.to(self.device)
                probs, aux = model(data)
                pred = probs.max(dim=1)[1]
                correct += pred.eq(data.nclose).sum().item()
        return correct / len(self.test_loader.dataset)  # does not account for drop_last

    def train(self, model, experiment_name):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 0.005 does well
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)
        with SummaryWriter(comment=experiment_name) as writer:
            print('Training', experiment_name)
            for epoch in range(1, self.EPOCHS + 1):
                losses, training_time = self.train_once(model, optimizer)
                test_acc = self.test(model)
                title = 'Epoch {:03d}, Total loss: {:.3f}, Test accuracy: {:.4f}, Duration {:.1f}s, LR: {:.5f}'.format(
                    epoch, losses["total"], test_acc, training_time, scheduler.get_lr()[0])
                print(title)
                if epoch % 10 == 0:
                    draw_n(16, 'Epoch {:03d}, Test accuracy: {:.2f}'.format(epoch, test_acc), model, device, self.test_loader.dataset, block=False)
                writer.add_scalars('Loss', losses, epoch)
                writer.add_scalar('Accuracy/predictions', test_acc, epoch)
                writer.add_scalar('Accuracy/score', test_acc * self.test_loader.dataset.num_nodes, epoch)
                scheduler.step()

    def benchmark(self, models):
        for experiment_name, model_fun in models.items():
            experiment_name += '_' + str(self.NUM_NODES).zfill(3)
            model = model_fun()
            self.train(model, experiment_name)
