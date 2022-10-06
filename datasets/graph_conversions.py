# NOTICE: 
# This file contains helper functions to construct superpixel and hierarchical 
# serpixel representation of individual images. This code taken from the 
# the work of Boris Knyazev, et al. which accompanies the piece
# "Image Classification with Hierarchical Multigraph Networks".
# It is redistributed in compliance with the Educational Community License, 
# Version 2.0 (ECL-2.0)
#
# GitHub Repository: https://github.com/bknyaz/bmvc_2019
# Paper: https://arxiv.org/abs/1907.09000

import numpy as np
import scipy.ndimage
import torch

from scipy.spatial.distance import cdist
from skimage.segmentation import slic
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

def sparsify_graph(A, knn_graph):
    if knn_graph is not None and knn_graph < A.shape[0]:
        idx = np.argsort(A, axis=0)[:-knn_graph, :]
        np.put_along_axis(A, idx, 0, axis=0)
        idx = np.argsort(A, axis=1)[:, :-knn_graph]
        np.put_along_axis(A, idx, 0, axis=1)
    return A

def spatial_graph(coord, img_size, knn_graph=32):
    coord = coord / np.array(img_size, np.float)
    dist = cdist(coord, coord)
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma**2)
    A[np.diag_indices_from(A)] = 0  # remove self-loops
    sparsify_graph(A, knn_graph)
    return A  # adjacency matrix (edges)

def superpixel_features(img, superpixels):
    n_sp = len(np.unique(superpixels))
    n_ch = img.shape[2]
    avg_values = np.zeros((n_sp, n_ch))
    coord = np.zeros((n_sp, 2))
    masks = []
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        for c in range(n_ch):
            avg_values[sp, c] = np.mean(img[:, :, c][mask])
        coord[sp] = np.array(scipy.ndimage.measurements.center_of_mass(mask))
        masks.append(mask)
    return avg_values, coord, masks

def preprocess_data(data):
    img = data[0]

    label = torch.tensor([data[1]])

    # convert it back to numpy because transform defaults to PIL and we're recycling code
    # but we also need to remember (coord, coord, #channels). Also slic() needs np.float64's
    img = np.array(img, dtype=np.float64)

    # normalize img
    img = (img / float(img.max()))

    # add a third dimension (if we're black and white image)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    else:
        img = img.transpose(1, 2, 0)

    return img, label

# passing a 1D array of number of segments to be consistent with prep_hier...
# will always take the first element so passing more won't actually do anything...
def prep_spatial(data, n_segments=[75], knn_graph=32):

    img, label = preprocess_data(data)

    # segment the image and put it back into
    superpixels = slic(img, n_segments=n_segments[0])

    avg_values, coord, masks = superpixel_features(img, superpixels)

    # keep only knn_graph neighbors for each node
    A_spatial = spatial_graph(coord, img.shape[:2], knn_graph=knn_graph)

    x = torch.from_numpy(avg_values).float()

    # convert A_spatial into edge indicies and edge attributes tensors
    edge_idx, edge_attr = dense_to_sparse(torch.from_numpy(A_spatial).float())
    # reshape edge_attr to [num_edges, num_features]
    edge_attr = edge_attr.unsqueeze(-1)

    pos = torch.from_numpy(coord).float()

    # return a PyTorch_Geometric Data object
    return Data(x=x,
                edge_index=edge_idx,
                edge_attr=edge_attr,
                pos=pos,
                y=label, num_nodes=x.shape[0])

def compute_iou_binary(seg1, seg2):
    inters = float(np.count_nonzero(seg1 & seg2))
    # areas can be precomputed in advance
    seg1_area = float(np.count_nonzero(seg1))
    seg2_area = float(np.count_nonzero(seg2))
    return inters / (seg1_area + seg2_area - inters)

def hierarchical_graph(masks_multiscale, n_sp_actual, knn_graph=32):
    n_sp_total = np.sum(n_sp_actual)
    A = np.zeros((n_sp_total, n_sp_total))
    for level1, masks1 in enumerate(masks_multiscale):
        for level2, masks2 in enumerate(masks_multiscale[level1+1:]):
            for i, mask1 in enumerate(masks1):
                for j, mask2 in enumerate(masks2):
                    A[np.sum(n_sp_actual[:level1], dtype=np.int) + i, np.sum(n_sp_actual[:level2+level1+1], dtype=np.int) + j] = compute_iou_binary(mask1, mask2)
    sparsify_graph(A, knn_graph)
    return A + A.T

def prep_hier(data, n_segments=[75, 21, 7], knn_graph=32):

    img, label = preprocess_data(data)

    # form the hierarchal portions
    n_sp_actual = []
    avg_values_multiscale, coord_multiscale, masks_multiscale = [], [], []
    # Scales [1000, 300, 150, 75, 21, 7] in the paper
    for i, sp in enumerate(n_segments):
        superpixels = slic(img, n_segments=sp)
        n_sp_actual.append(len(np.unique(superpixels)))
        avg_values_, coord_, masks_ = superpixel_features(img, superpixels)
        avg_values_multiscale.append(avg_values_)
        coord_multiscale.append(coord_)
        masks_multiscale.append(masks_)
    A_spatial_multiscale = spatial_graph(np.concatenate(coord_multiscale), img.shape[:2], knn_graph=knn_graph)
    A_hier = hierarchical_graph(masks_multiscale, n_sp_actual, knn_graph=None)

    x = torch.from_numpy(np.concatenate(avg_values_multiscale)).float()

    stacked = torch.from_numpy(np.stack((A_spatial_multiscale, A_hier), axis=2)).float()

    edge_index = stacked.nonzero().t().contiguous()
    edge_attr = stacked[edge_index[0], edge_index[1]]

    # slice out unnecessary indicies
    edge_index = edge_index[:2,:]

    pos = torch.from_numpy(np.concatenate(coord_multiscale)).float()

    # return a PyTorch_Geometric Data object
    return Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                y=label,
                num_nodes=x.shape[0])
