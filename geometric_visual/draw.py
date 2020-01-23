import math
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch as th

from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx


def draw_image_graph(image, G, center, colors=None, target_color=None, alpha=0.5, block=True, title='Geometric Visual', fig=0, ax=None):
    width, height, _ = image.shape
    if colors is None:
        colors = np.random.random((len(G.nodes), 3))
        if target_color is not None:
            colors = colors * np.array(target_color)
    pos = nx.get_node_attributes(G, 'pos')
    pos_large = {node: (width * p[0], height * p[1]) for node, p in pos.items()}
    ax_or_plt = plt if ax is None else ax
    plt.figure(fig)
    ax_or_plt.cla()
    ax_or_plt.imshow(
        np.transpose(image, axes=(1, 0, 2)),
        aspect='equal',
        interpolation='none',
        origin='lower',  # The convention 'upper' is typically used for matrices and images
        alpha=alpha
    )
    nx.draw_networkx(G, pos=pos_large, edge_color=(1., 1., 1.), node_color=colors, ax=ax)
    ax_or_plt.scatter(center[0] * width, center[1] * height, c='k')
    #plt.axis('on')
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.05)


def _data_to_nx(data, out=None, aux=None):
    data = data.to('cpu')
    out = out.to('cpu') if out is not None else out
    aux = aux.to('cpu') if aux is not None else aux
    G = to_networkx(data, node_attrs=None, edge_attrs=None)
    pos = {node: p for node, p in zip(G.nodes, np.array(data.x))}
    nx.set_node_attributes(G, pos, name='pos')
    image = np.array(data.im.squeeze(), dtype=np.float64).transpose(2, 1, 0)

    # chose colors from white to target_color
    color = 1 - data.target_color
    colors = th.stack([color[0]] * len(pos))
    target = data.y if out is None else out
    colors = 1 - target.unsqueeze(1) * colors
    center = data.square_center.squeeze() if aux is None else aux.squeeze()
    return image, G, center, colors


def draw_data(data, out=None, aux=None, block=False, title='', ax=None):
    image, G, center, colors = _data_to_nx(data, out, aux)
    title = 'Data with %ith closest node(s): %s' % (data.nclose.item() + 1, data.nclosest.tolist()) if not title else title
    draw_image_graph(image, G, center, colors=np.array(colors, dtype=np.float64), block=block, title=title, ax=ax)


def draw_data_list(data_list, out_list=None, aux_list=None, fig=0, title='', block=False, alpha=0.5):
    n = len(data_list)
    nrows = int(math.sqrt(n) + 0.99)
    ncols = max(1, int(n/nrows + 0.99))  # if it works...
    # subpots is too slow and ugly result
    # combine all in one image
    axes = product(range(nrows), range(ncols))
    _, _, height, width = data_list[0].im.size()
    combined_image = np.ones(((width+1) * ncols - 1, (height+1) * nrows - 1, 3), dtype=np.float64)
    G_list = []
    out_list = [None] * n if out_list is None else out_list
    aux_list = [None] * n if aux_list is None else aux_list
    for data, out, aux, ax in zip(data_list, out_list, aux_list, axes):
        image, G, center, colors = _data_to_nx(data, out, aux)
        slice_width = (width*ax[1]+ax[1], width*(ax[1]+1)+ax[1])
        slice_height = (height*ax[0]+ax[0], height*(ax[0]+1)+ax[0])
        combined_image[slice_width[0]:slice_width[1], slice_height[0]:slice_height[1], :] = image
        pos = nx.get_node_attributes(G, 'pos')
        pos_large = {node: (width * p[0] + slice_width[0], height * p[1] + slice_height[0]) for node, p in pos.items()}
        center_large = slice_width[0] + center[0] * width, slice_height[0] + center[1] * height
        G_list.append((G, pos_large, center_large, colors))
    plt.figure(fig)
    plt.cla()
    plt.imshow(
        np.transpose(combined_image, axes=(1, 0, 2)),
        aspect='equal',
        interpolation='none',
        origin='lower',  # The convention 'upper' is typically used for matrices and images
        alpha=alpha
    )
    for G, pos_large, center_large, colors in G_list:
        nx.draw_networkx(G, pos=pos_large, edge_color=(1., 1., 1.), node_color=colors, node_size=400//nrows, font_size=8, width=1.2)
        plt.scatter(center_large[0], center_large[1], c='k', s=10)
    plt.xlim(0, combined_image.shape[0])
    plt.ylim(0, combined_image.shape[1])
    plt.title(title)
    plt.show(block=block)
    plt.pause(0.05)


def draw_n(n, title, model, model_device, dataset, block=False):
    with th.no_grad():
        batch = Batch.from_data_list(list(dataset.generate_n_data(n))).to(model_device)
        probs, aux = model(batch)
    data_list = batch.to_data_list()
    out_list = [probs[i] for i in range(n)]
    aux_list = [aux[i] for i in range(n)]
    draw_data_list(data_list, out_list=out_list, aux_list=aux_list, block=block, title=title)


if __name__ == '__main__':
    from .dataset import GeoVisualDataset
    dataset = GeoVisualDataset(16, 10, 3, 64, 64, (1.,0.,0.), radius=0.3)
    data_list = list(dataset)
    draw_data_list(data_list, block=True)
