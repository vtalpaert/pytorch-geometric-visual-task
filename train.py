from time import process_time

import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader, Batch
import torch_geometric.transforms as T

from geometric_visual.dataset import GeoVisualDataset
from geometric_visual.model import GCNNet
from geometric_visual.draw import draw_n

IM_CHANNELS = 3
IM_HEIGHT = IM_WIDTH = 64
NUM_NODES = 20
RADIUS = 0.3
MAX_NUM_NEIGHBORS = 3
TARGET_COLOR = (1.,0.,0.)
BATCH_SIZE = 64
AUXILIARY_TASK_WEIGHT = 1.0
NUM_WORKERS = 8
SIZE_TRAIN = 8000
SIZE_TEST = 2000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = GeoVisualDataset(SIZE_TRAIN, NUM_NODES, IM_HEIGHT, IM_WIDTH, TARGET_COLOR, radius=RADIUS, max_num_neighbors=MAX_NUM_NEIGHBORS, device='cpu')
test_dataset = GeoVisualDataset(SIZE_TEST, NUM_NODES, IM_HEIGHT, IM_WIDTH, TARGET_COLOR, radius=RADIUS, max_num_neighbors=MAX_NUM_NEIGHBORS, device='cpu')
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)

model = GCNNet(channels=IM_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()
    start = process_time()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        probs, aux = model(data)
        loss_probs = F.binary_cross_entropy(probs, data.y.view(BATCH_SIZE, NUM_NODES).type(dtype=torch.float)) #F.cross_entropy(scores, data.nclose)  # F.binary_cross_entropy_with_logits
        loss_aux = F.mse_loss(aux, data.square_center)
        loss = loss_probs + AUXILIARY_TASK_WEIGHT * loss_aux
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    end = process_time()
    return total_loss / len(train_dataset), end - start


def test(epoch):
    model.eval()

    correct = 0
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            data = data.to(device)
            probs, aux = model(data)
            #probs = F.sigmoid(scores)
            pred = probs.max(dim=1)[1]
            correct += pred.eq(data.nclose).sum().item()
    return correct / len(test_loader.dataset)  # does not account for drop_last


for epoch in range(1, 201):
    loss, training_time = train()
    test_acc = test(epoch)
    title = 'Epoch {:03d}, Loss: {:.3f}, Test accuracy: {:.2f}, Duration {:.1f}s, LR: {:.5f}'.format(
        epoch, loss, test_acc, training_time, scheduler.get_lr()[0])
    print(title)
    draw_n(16, title, model, device, test_dataset, block=epoch > 190)
    scheduler.step()
