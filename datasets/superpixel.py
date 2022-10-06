import torch
from torch_geometric import datasets
import torchvision.transforms as T

from scipy.spatial import Delaunay
from torch_geometric.data import InMemoryDataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from tqdm import tqdm
from datasets.graph_conversions import prep_hier, prep_spatial
from pathlib import Path
from filelock import FileLock


def get_datasets(name, hierarchical, prune_edges):
    home = Path(__file__).parent.parent
    d_path =  home / 'data'
    d_name = name
    if hierarchical:
        d_name += '-Hierarchical'
    else:
        d_name += '-SP'
    if prune_edges:
        d_name += '-Pruned'

    if name == "MNIST":
        if hierarchical:
            dataset_func = MNISTHierarchical
        else:
            dataset_func = MNISTSuperpixel
    elif name == "CIFAR10":
        dataset_func = CIFAR10Superpixel
    elif name == "CIFAR100":
        dataset_func = CIFAR100Superpixel
    else:
        raise ValueError("Unknown dataset")

    d_path = d_path / d_name
    d_path.mkdir(parents=True, exist_ok=True)

    with FileLock(d_path / 'train_data.lock'):
        train_dataset = dataset_func(
            d_path, train=True, hierarchical=hierarchical,
            prune=prune_edges
        )
    with FileLock(d_path / 'test_data.lock'):
        test_dataset = dataset_func(
            d_path, train=False, hierarchical=hierarchical,
            prune=prune_edges
        )
    return train_dataset, test_dataset


class CIFAR10Superpixel(InMemoryDataset):
    def __init__(self, root, train=True, hierarchical=False,
                 transform=None, pre_transform=None, prune=False,
                 knn_graph=32):

        self.train = train
        self.prune = prune

        if hierarchical:
            self.prefix = 'hier_'
            self.graph_func = prep_hier
        else:
            self.prefix = 'spatial_'
            self.graph_func = prep_spatial

        if train:
            self.prefix += 'train'
        else:
            self.prefix += 'test'

        self.knn_graph = knn_graph

        super(CIFAR10Superpixel, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['CIFAR10/train.pt', 'CIFAR10/test.pt']

    @property
    def processed_file_names(self):
        return [self.prefix+'.pt']

    def download(self):
        pass

    def process(self):
        dataset = CIFAR10(self.raw_dir, train=self.train, download=True, transform=T.Compose([T.ToTensor()]))

        data_list = []

        for d in tqdm(dataset, desc='Processed Graphs', total=len(dataset)):
            data = self.graph_func(d, n_segments=[150, 75, 21, 7], knn_graph=self.knn_graph)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CIFAR100Superpixel(InMemoryDataset):
    def __init__(self, root, train=True, hierarchical=False, transform=None, pre_transform=None, prune=False, knn_graph=32):

        self.train = train
        self.prune = prune

        if hierarchical:
            self.prefix = 'hier_'
            self.graph_func = prep_hier
        else:
            self.prefix = 'spatial_'
            self.graph_func = prep_spatial

        if train:
            self.prefix += 'train'
        else:
            self.prefix += 'test'

        self.knn_graph = knn_graph

        super(CIFAR100Superpixel, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['CIFAR100/train.pt', 'CIFAR100/test.pt']

    @property
    def processed_file_names(self):
        return [self.prefix+'.pt']

    def download(self):
        pass

    def process(self):
        dataset = CIFAR100(self.raw_dir, train=self.train, download=True, transform=T.Compose([T.ToTensor()]))

        data_list = []

        for d in tqdm(dataset, desc='Processed Graphs', total=len(dataset)):
            data = self.graph_func(d, n_segments=[150, 75, 21, 7], knn_graph=self.knn_graph)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def MNISTSuperpixel(root, train=True, hierarchical=True, transform=None, pre_transform=None, prune=False, knn_graph=32):
    return datasets.MNISTSuperpixels(root, train=train, transform=transform,
                              pre_transform=pre_transform)


class MNISTHierarchical(InMemoryDataset):
    def __init__(self, root, train=True, hierarchical=True, transform=None, pre_transform=None, prune=False, knn_graph=32):

        self.train = train
        self.prune = prune

        if hierarchical:
            self.prefix = 'hier_'
            self.graph_func = prep_hier
        else:
            self.prefix = 'spatial_'
            self.graph_func = prep_spatial

        if train:
            self.prefix += 'train'
        else:
            self.prefix += 'test'

        self.knn_graph = knn_graph

        super(MNISTHierarchical, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['MNIST/processed/training.pt', 'MNIST/processed/test.pt']

    @property
    def processed_file_names(self):
        return [self.prefix+'.pt']

    def download(self):
        pass

    def process(self):
        dataset = MNIST(self.raw_dir, train=self.train, download=True, transform=T.Compose([T.ToTensor()]))

        data_list = []

        for d in tqdm(dataset, desc='Processed Graphs', total=len(dataset)):
            data = self.graph_func(d,  n_segments=[75, 21, 7], knn_graph=self.knn_graph)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
