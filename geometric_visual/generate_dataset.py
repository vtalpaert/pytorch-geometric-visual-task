import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_random_image(height, width, square_size, square_color=(1.,0.,0.)):
    #image = np.zeros((height, width, 3))
    image = np.random.random((height, width, 3))
    x = np.random.randint(0, height - square_size)
    y = np.random.randint(0, width - square_size)
    image[x:x+square_size,y:y+square_size] = square_color
    return image

def generate_random_graph(n, radius, height, width, seed=None):
    G = nx.random_geometric_graph(n, radius)
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(G, 'pos')
    for node, p in pos.items():
        pos[node] = [0.90 * (p[0] + 0.05) * height, 0.90 * (p[1] + 0.05) * width]
    return G, pos

def plot(image, G, pos, colors=None):
    if colors is None:
        colors = list(G.nodes)
    plt.figure()
    plt.imshow(
        image,
        alpha=0.5
    )
    nx.draw_networkx(G, pos, edge_color=(1., 1., 1.), node_color=colors, cmap=plt.cm.Reds_r)
    plt.axis('on')
    plt.show()

if __name__ == "__main__":
    height, width = 64, 64
    image = generate_random_image(height, width, 5)
    G, pos = generate_random_graph(100, 0.125, height, width)
    plot(image, G, pos)
