import numpy as np
import torch as th
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from torch_geometric.nn import radius_graph


def generate_random_image_th(height, width, square_size, square_color, batch_size, device):
    assert square_size > 1
    im = th.rand(3, height, width, device=device)
    square_x = th.randint(width - square_size, (1,), device=device)
    square_y = th.randint(height - square_size, (1,), device=device)
    color = square_color
    square = th.cat([th.cat([color.transpose(0, 1)] * square_size, dim=1).unsqueeze_(2)] * square_size, dim=2)
    im[0:3, square_y:square_y+square_size, square_x:square_x+square_size] = square
    x_normalized = (square_x.item() + square_size // 2) / width
    y_normalized = (square_y.item() + square_size // 2) / height
    square_center = th.tensor((x_normalized, y_normalized), device=device).resize_(1, 2)
    return im.unsqueeze_(0), square_center

def generate_random_graph(n, radius, max_num_neighbors, batch_size, device):
    # arrange for points to fit more inside the unit square
    batch = None
    num_nodes = th.tensor([n], device=device, dtype=th.long)
    x = 0.90 * (th.rand(n, 2, device=device) + 0.05)
    edge_index = radius_graph(x, radius, batch=batch, loop=True, max_num_neighbors=max_num_neighbors)  # COO format
    return x, edge_index, num_nodes, batch

def generate_groundtruth_th(center, x, nclose, batch, device):
    d = th.sum((x - center) ** 2, dim=1)
    nclosest = th.argsort(d, dim=0, descending=False)[nclose]
    one_hot = F.one_hot(nclosest, x.size(0))
    return one_hot, nclosest.unsqueeze_(0)  # one_hot.type(dtype=th.float)

def generate_data(n, nclose, height, width, target_color, batch_size=1, device='cpu', square_size=5, radius=0.125, max_num_neighbors=5):
    """Creates Data object
    Inputs:
        - n (int): number of points in the graph
        - nclose (int): index of the closest point to find, first closest is nclose=0, second nclose=1
        - height, widht (ints): image pixel size
        - target_color (tuple): float color RGB embeded in the image as a square
        - batch_size (int): TODO return a Batch (Data if batch_size is 1)
        - device (str or torch device): device for data, use only cpu for now as long as batch creation is not done
        - square_size (int): pixel size of square in image
        - radius (float): value for edge generation based on closed neighbours between 0 and 1
        - max_num_neighbors: maximum number of edges for graph creation
    
    Output:
        - Data object with attributes:
            - edge_index (LongTensor): [2, num_edges] connections between nodes
            - x (FloatTensor): [num_nodes, 2] the 2D normalized positions between 0 and 1
            - y (LongTensor): [num_nodes] node level target (use .type(dtype=torch.float) for FloatTensor)
            - nclosest (LongTensor): [batch_size] contains the node index of the nclose-th closest
            - num_nodes_per_graph (LongTensor): [batch_size] contains the number of nodes in the batch element
            - im (tensor): in PyTorch style [batch_size, 3, height, width]
            - target_color (tensor): [batch_size, 3]
            - square_center (tensor): [batch_size, 2]

    The are no edge features (data.edge_attr)
    """
    if batch_size > 1:
        raise NotImplementedError("Creating a Batch directly is not implemented yet, use DataLoader to collate Data objects")
    target_color = th.tensor([target_color], device=device)
    nclose_th = th.tensor([nclose], dtype=th.int, device=device)
    x, edge_index, num_nodes, batch = generate_random_graph(n, radius, max_num_neighbors, batch_size, device)
    im, square_center = generate_random_image_th(height, width, square_size, target_color, batch_size, device)
    y, nclosest = generate_groundtruth_th(square_center, x, nclose, batch, device)
    data = Data.from_dict({
        'x': x,
        'edge_index': edge_index,
        'y': y,
        'nclose': nclose_th,
        'nclosest': nclosest,
        'num_nodes_per_graph': num_nodes,
        'im': im,
        'target_color': target_color,
        'square_center': square_center
    })
    return data


if __name__ == "__main__":
    height, width = 32, 64
    target_color = (1.,0.,0.)
    n = 100
    nclose = 2
    radius = 0.125
    max_num_neighbors = 5
    data = generate_data(n, nclose, height, width, target_color, batch_size=1, radius=radius, max_num_neighbors=max_num_neighbors)
    print('data.x', data.x.type(), data.x.size())
    print('data.edge_index', data.edge_index.type(), data.edge_index.size())
    print('data.y', data.y.type(), data.y.size())
    print(data)
    from .draw import draw_data
    draw_data(data, block=True)
