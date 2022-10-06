from torch_geometric import datasets
import torch_geometric.transforms as T
from datasets.superpixel import MNISTSuperpixel, MNISTHierarchical, CIFAR10Superpixel, CIFAR100Superpixel
from pathlib import Path
from filelock import FileLock


def get_datasets(name, hierarchical, prune_edges):
    home = Path(__file__).parent
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

    with FileLock(d_path / 'train_data.lock'):
        train_dataset = dataset_func(
            d_path / d_name, train=True, hierarchical=hierarchical,
            prune=prune_edges
        )
    with FileLock(d_path / 'test_data.lock'):
        test_dataset = dataset_func(
            d_path / d_name, train=False, hierarchical=hierarchical,
            prune=prune_edges
        )
    return train_dataset, test_dataset
