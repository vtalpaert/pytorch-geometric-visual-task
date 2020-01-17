from time import process_time
from math import log

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from geometric_visual.draw import draw_n


def train(model, train_loader, optimizer, device, auxiliary_task_weight=0):
    model.train()
    start = process_time()
    losses = {
            "probs": 0,
            "aux": 0,
            "total": 0
        }
    N = train_loader.dataset.num_nodes
    loss_probs_norm = (log(N)-(N-1)*log(1-1/N))/N
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        probs, aux = model(data)
        target = data.y.view(data.num_graphs, -1).type(dtype=torch.float)
        num_nodes = target.size(1)
        loss_probs = F.binary_cross_entropy(probs, target, reduction='mean') #F.cross_entropy(scores, data.nclose)  # F.binary_cross_entropy_with_logits
        loss_probs /= loss_probs_norm
        loss_aux = F.mse_loss(aux, data.square_center, reduction='mean')
        loss_aux *= 4.5
        loss =  (loss_probs + auxiliary_task_weight * loss_aux) / (1 + auxiliary_task_weight)
        loss.backward()
        losses["total"] += data.num_graphs * loss.item()
        losses["probs"] += data.num_graphs * loss_probs.item()
        losses["aux"] += data.num_graphs * loss_aux.item()
        optimizer.step()
    end = process_time()
    size = train_loader.dataset.size
    losses["total"] /= size
    losses["probs"] /= size
    losses["aux"] /= size
    return losses, end - start


def test(model, test_loader, device):
    model.eval()

    correct = 0
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            data = data.to(device)
            probs, aux = model(data)
            pred = probs.max(dim=1)[1]
            correct += pred.eq(data.nclose).sum().item()
    return correct / len(test_loader.dataset)  # does not account for drop_last


def benchmark(model, train_loader, test_loader, auxiliary_task_weight, comment='', max_epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 0.005 does well
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)
    with SummaryWriter(comment=comment) as writer:
        print('Training', comment)
        for epoch in range(1, max_epochs + 1):
            losses, training_time = train(model, train_loader, optimizer, device, auxiliary_task_weight=auxiliary_task_weight)
            test_acc = test(model, test_loader, device)
            title = 'Epoch {:03d}, Total loss: {:.3f}, Test accuracy: {:.4f}, Duration {:.1f}s, LR: {:.5f}'.format(
                epoch, losses["total"], test_acc, training_time, scheduler.get_lr()[0])
            print(title)
            if epoch % 10 == 0:
                draw_n(16, 'Epoch {:03d}, Test accuracy: {:.2f}'.format(epoch, test_acc), model, device, test_loader.dataset, block=False)
            writer.add_scalars('Loss', losses, epoch)
            writer.add_scalar('Accuracy/predictions', test_acc, epoch)
            writer.add_scalar('Accuracy/score', test_acc * test_loader.dataset.num_nodes, epoch)
            scheduler.step()
