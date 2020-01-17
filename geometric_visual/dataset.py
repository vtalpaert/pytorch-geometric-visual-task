from torch.utils.data.dataset import IterableDataset
from torch.utils.data import get_worker_info
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from .data import generate_data


class GeoVisualDataset(IterableDataset):
    def __init__(self, size, *args, transform=None, **kwargs):
        super(GeoVisualDataset).__init__()
        self.size = size
        self.transform = transform
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return self.size

    @property
    def num_nodes(self):
        return self.args[0]  # first arg of data.generate_data

    def generate_data(self):
        if self.transform is not None:
            return self.transform(generate_data(*self.args, **self.kwargs))
        else:
            return generate_data(*self.args, **self.kwargs)

    def generate_n_data(self, n):
        return (self.generate_data() for _ in range(n))  # generator

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            size = self.size
        else:  # in a worker process
            # split workload
            size = int(self.size / float(worker_info.num_workers))
        datalist = self.generate_n_data(size)
        return datalist


def generate_dataloader(dataset_size, batch_size, *args, transform=None, **kwargs):
    kwargs['transform'] = transform
    return DataLoader(GeoVisualDataset(dataset_size, *args, **kwargs), batch_size=batch_size)


class AddNodeViews(object):
    """Add a crop view of the image to each node"""
    # TODO


if __name__ == "__main__":
    height, width = 32, 64
    target_color = (1.,0.,0.)
    loader = generate_dataloader(1000, 8, 100, height, width, target_color)
    batch = next(iter(loader))
    print(batch)
    data0, data1 = batch.to_data_list()[0:2]
    print(data0)
    print(data1.edge_index)
    next(iter(loader))
