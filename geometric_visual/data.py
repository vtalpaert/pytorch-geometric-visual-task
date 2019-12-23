import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx, to_networkx


def generate_random_image(height, width, square_size, square_color):
    assert square_size > 1
    #image = np.zeros((width, height, 3))
    image = np.random.random((width, height, 3))
    square_x = np.random.randint(0, width - square_size)
    square_y = np.random.randint(0, height - square_size)
    image[square_x:square_x+square_size,square_y:square_y+square_size] = square_color
    x_normalized = (square_x + square_size / 2) / width
    y_normalized = (square_y + square_size / 2) / height
    return image, (x_normalized, y_normalized)

def generate_random_graph(n, radius, height, width, seed=None):
    G = nx.random_geometric_graph(n, radius)
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(G, 'pos')
    for node, p in pos.items():
        # arrange for points to fit more inside the unit square
        pos[node] = [0.90 * (p[0] + 0.05), 0.90 * (p[1] + 0.05)]
    nx.set_node_attributes(G, pos, name='pos')
    return G, pos

def generate_groundtruth(sx, sy, pos):
    # find node near square center (sx, sy)
    d = np.sum(np.square(np.array(list(pos.values())) - np.array((sx, sy))), axis=1)
    nclose = np.argmin(d)
    one_hot = np.zeros(len(pos))
    one_hot[nclose] = 1
    return one_hot, nclose, np.sqrt(d[nclose])

def generate_data(G, y, image, target_color, square_center=None, use_float32=True):
    """Creates Data object
    Inputs:
        - G (networkx graph): graph with pos attribute
        - y (list or array): one hot encoded target in [num_nodes (, 1)]
        - image (array): in [width, height, 3]
        - target_color (tuple): float color RGB
        - square center: not used
        - use_float32: whether to use float16 or float32 type
    
    Output:
        - Data object with attributes:
            - edge_index (tensor): [2, num_edges] connections between nodes
            - pos (tensor): [num_nodes, 2] the 2D normalized positions
            - y (tensor): [num_nodes, 1] node level target
            - im (tensor): in PyTorch style [3, height, width]
            - target_color (tensor): [3]

    The are no node or edge features (data.x, data.edge_attr)
    """
    dtype = np.float32 if use_float32 else np.float64
    data = from_networkx(G)
    y = np.array(y, dtype=dtype)
    y = th.tensor(y).reshape((-1, 1))
    data.y = y
    #data.image = image
    data.im = th.tensor(image.astype(dtype).transpose(2, 1, 0))
    data.target_color = th.FloatTensor(target_color) if use_float32 else th.DoubleTensor(target_color)
    data.square_center = np.array(square_center, dtype=dtype)
    return data

def generate_dataloader():
    data_list = []
    loader = DataLoader(data_list, batch_size=32)

def draw_data(data, image_ref, out=None):  # TODO
    G = to_networkx(data, node_attrs=None, edge_attrs=None)
    pos = {node: p for node, p in zip(G.nodes, np.array(data.pos))}
    nx.set_node_attributes(G, pos, name='pos')
    image = np.array(data.im, dtype=np.float64).transpose(2, 1, 0)
    
    # chose colors from white to target_color
    color = 1 - np.array(data.target_color, dtype=np.float64)
    colors = np.stack([color] * len(pos))
    target = np.array(data.y, dtype=np.float64) if out is None else np.array(out, dtype=np.float64)
    colors = 1 - colors * target
    draw_image_graph(image, G, colors=colors)

def draw_image_graph(image, G, colors=None, target_color=None, alpha=0.5):
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
    print('pos', pos[0], 'pos_large', pos_large[0])
    nx.draw_networkx(G, pos=pos_large, edge_color=(1., 1., 1.), node_color=colors)
    plt.axis('on')
    plt.show()

if __name__ == "__main__":
    height, width = 32, 64
    target_color = (1.,0.,0.)
    image, (sx, sy) = generate_random_image(height, width, square_size=5, square_color=target_color)
    G, pos = generate_random_graph(100, 0.125, height, width)
    y, nclose, dmin = generate_groundtruth(sx, sy, pos)
    #draw_image_graph(image, G, target_color=(0, 1, 0))
    data = generate_data(G, y, image, target_color, (sx, sy))
    draw_data(data, image)
