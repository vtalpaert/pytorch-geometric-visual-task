import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import radius_graph


def generate_random_image_np(height, width, square_size, square_color):
    assert square_size > 1
    image = np.random.random((width, height, 3))
    square_x = np.random.randint(0, width - square_size)
    square_y = np.random.randint(0, height - square_size)
    image[square_x:square_x+square_size,square_y:square_y+square_size] = square_color
    x_normalized = (float(square_x) + square_size // 2) / width
    y_normalized = (float(square_y) + square_size // 2) / height
    return image, (x_normalized, y_normalized)

def generate_random_image_th(height, width, square_size, square_color, batch):
    assert square_size > 1
    im = th.rand(3, height, width)
    square_x = th.randint(width - square_size, (1,))
    square_y = th.randint(height - square_size, (1,))
    square = th.cat([th.cat([square_color.transpose(0, 1)] * square_size, dim=1).unsqueeze_(2)] * square_size, dim=2)
    im[:, square_y:square_y+square_size, square_x:square_x+square_size] = square
    x_normalized = (float(square_x) + square_size // 2) / width
    y_normalized = (float(square_y) + square_size // 2) / height
    square_center = th.tensor((x_normalized, y_normalized)).resize_(1, 2)
    if batch:
        im = im.unsqueeze_(0)
        square_center = square_center.unsqueeze_(0)
    return im, square_center

def generate_random_networkx(n, radius, seed=None):
    G = nx.random_geometric_graph(n, radius, seed=seed)
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(G, 'pos')
    for node, p in pos.items():
        # arrange for points to fit more inside the unit square
        pos[node] = [0.90 * (p[0] + 0.05), 0.90 * (p[1] + 0.05)]
    nx.set_node_attributes(G, pos, name='pos')
    return G, pos

def generate_random_graph(n, radius, batch):
    # arrange for points to fit more inside the unit square
    x = 0.90 * (th.rand(n, 2) + 0.05)
    edge_index = radius_graph(x, radius)
    if batch:
        x, edge_index = x.unsqueeze_(0), edge_index.unsqueeze_(0)
    return x, edge_index

def generate_groundtruth_np(sx, sy, pos):
    # find node near square center (sx, sy)
    d = np.sum(np.square(np.array(list(pos.values())) - np.array((sx, sy))), axis=1)
    nclose = np.argmin(d)
    one_hot = np.zeros(len(pos))
    one_hot[nclose] = 1
    return one_hot, nclose, np.sqrt(d[nclose])

def generate_groundtruth_th(center, x, batch):
    dim = 2 if batch else 1
    d = th.sum((x - center) ** 2, dim=dim)
    nclose = th.argmin(d, dim=dim-1)
    one_hot = F.one_hot(nclose, x.size(dim-1))  # [batch, num_nodes] or [num_nodes]
    return one_hot

def generate_data(n, height, width, target_color, batch, square_size=5, radius=0.125, use_float32=True):
    """Creates Data object
    Inputs:
        - n (int): number of points in the graph
        - height, widht (ints): image pixel size
        - target_color (tuple): float color RGB embeded in the image as a square
        - batch (bool): wether to unsqueeze in the first dim
        - square_size (int): pixel size of square in image
        - radius (float): value for edge generation based on closed neightbours between 0 and 1
        - use_float32: whether to use float32 or float64 type
    
    Output:
        - Data object with attributes:
            - edge_index (tensor): [2, num_edges] connections between nodes
            - x (tensor): [num_nodes, 2] the 2D normalized positions between 0 and 1
            - y (tensor): [num_nodes] node level target
            - im (tensor): in PyTorch style [3, height, width]
            - target_color (tensor): [1, 3]
            - square_center (tensor): [1, 2]

    The are no edge features (data.edge_attr)
    """
    Tensor = th.FloatTensor if use_float32 else th.DoubleTensor
    target_color = Tensor(target_color).resize_(1, 3)
    if batch:
        target_color = target_color.unsqueeze_(0)
    x, edge_index = generate_random_graph(n, radius, batch)
    im, square_center = generate_random_image_th(height, width, square_size, target_color, batch)
    y = generate_groundtruth_th(square_center, x, batch)
    return Data.from_dict({
        'x': x,
        'edge_index': edge_index,
        'y': y,
        'im': im,
        'target_color': target_color,
        'square_center': square_center
    })

def draw_data(data, out=None):
    G = to_networkx(data, node_attrs=None, edge_attrs=None)
    pos = {node: p for node, p in zip(G.nodes, np.array(data.x))}
    nx.set_node_attributes(G, pos, name='pos')
    image = np.array(data.im.squeeze(), dtype=np.float64).transpose(2, 1, 0)

    # chose colors from white to target_color
    color = 1 - data.target_color
    colors = th.stack([color[0]] * len(pos))
    target = data.y if out is None else out
    colors = 1 - target.unsqueeze(1) * colors
    draw_image_graph(image, G, data.square_center.squeeze(), colors=np.array(colors, dtype=np.float64))

def draw_image_graph(image, G, center, colors=None, target_color=None, alpha=0.5):
    width, height, _ = image.shape
    if colors is None:
        colors = np.random.random((len(G.nodes), 3))
        if target_color is not None:
            colors = colors * np.array(target_color)
    pos = nx.get_node_attributes(G, 'pos')
    pos_large = {node: (width * p[0], height * p[1]) for node, p in pos.items()}
    plt.figure()
    plt.imshow(
        np.transpose(image, axes=(1, 0, 2)),
        aspect='equal',
        interpolation='none',
        origin='lower',  # The convention 'upper' is typically used for matrices and images
        alpha=alpha
    )
    nx.draw_networkx(G, pos=pos_large, edge_color=(1., 1., 1.), node_color=colors)
    plt.scatter(center[0] * width, center[1] * height, c='k')
    plt.axis('on')
    plt.show()

if __name__ == "__main__":
    height, width = 32, 64
    target_color = (1.,0.,0.)
    n = 100
    radius = 0.125
    data = generate_data(n, height, width, target_color, batch=False, radius=radius)
    draw_data(data)
