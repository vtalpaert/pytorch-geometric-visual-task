import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, DenseGCNConv, GCNConv
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from geometric_visual.dataset import GeoVisualDataset
from geometric_visual.data import draw_data

IM_CHANNELS = 3
IM_HEIGHT = IM_WIDTH = 64
NUM_NODES = 20
RADIUS=0.2
TARGET_COLOR = (1.,0.,0.)
BATCH_SIZE = 64
NUM_WORKERS = 6

train_dataset = GeoVisualDataset(80000, NUM_NODES, IM_HEIGHT, IM_WIDTH, TARGET_COLOR, radius=RADIUS)
test_dataset = GeoVisualDataset(2000, NUM_NODES, IM_HEIGHT, IM_WIDTH, TARGET_COLOR, radius=RADIUS)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)


class LinView(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size

    def forward(self, x):
        return x.view(-1, self.feature_size)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_features = nn.Sequential(
            nn.Conv2d(IM_CHANNELS, 10, 11, stride=4, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(10, 16, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            LinView(64),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        #self.densegconv = DenseGCNConv(4, 1, bias=False)
        self.gconv = GCNConv(4, 1, bias=True, improved=True)
        self.gconv.aggr = 'mean'

    def forward(self, data):
        batch_im = data.im.view(BATCH_SIZE, IM_CHANNELS, IM_HEIGHT, IM_WIDTH)
        image_features = self.global_features(batch_im)
        repeated_image_features = torch.repeat_interleave(image_features, NUM_NODES, dim=0)  # BN x 2
        x = torch.cat([data.x, repeated_image_features], dim=1)  # BN x 4
        #adj = to_dense_adj(data.edge_index, batch=data.batch)
        #graph_features = self.densegconv(x, adj, add_loop=True)
        graph_features = F.leaky_relu_(self.gconv(x, data.edge_index))  # self.gconv2(F.leaky_relu_(self.gconv1(x, data.edge_index)), data.edge_index)
        out = graph_features.view(BATCH_SIZE, NUM_NODES)
        return F.softmax(out, dim=1), image_features  # F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        probs, aux = model(data)
        loss_probs = F.binary_cross_entropy(probs, data.y.view(BATCH_SIZE, NUM_NODES).type(dtype=torch.float)) #F.cross_entropy(scores, data.nclose)  # F.binary_cross_entropy_with_logits
        loss_aux = F.mse_loss(aux, data.square_center)
        loss = loss_probs + loss_aux
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(epoch):
    model.eval()

    correct = 0
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            probs, aux = model(data)
            #probs = F.sigmoid(scores)
            pred = probs.max(dim=1)[1]
            correct += pred.eq(data.nclose).sum().item()
        if i == 0:
            draw_data(data.to_data_list()[0].to('cpu'), out=probs[0].to('cpu'), aux=aux[0].to('cpu'), block=epoch > 190)
        
    return correct / len(test_loader.dataset)  # does not account for drop_last


for epoch in range(1, 201):
    loss = train()
    test_acc = test(epoch)
    print('Epoch {:03d}, Loss: {:.4f}, Test accuracy: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()