import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import max_pool, voxel_grid, knn_interpolate
import torch_geometric.transforms as T


triangulate = T.Compose([T.Delaunay(), T.FaceToEdge()])


class DPOConv(torch_geometric.nn.MessagePassing):
    '''
    Performs 2D convolution using PDO method.

    Parameters:
        in_channels: number of input channels
        out_channels: number of output channels
        r_correction: factor to adapt r by (should be such that the r value is
                      around 1)

        returns nodes with out_channels channels for each
    '''
    def __init__(self, in_channels, out_channels, r_correction=28):
        super(DPOConv, self).__init__(aggr='mean')
        self.r_correction = r_correction

        # every input channel has 4 items calculated for it before going
        # through neural network
        self.lin = nn.Linear(4 * in_channels, out_channels)

    def forward(self, x, edge_index, pos):
        # calculate phi (passes values to the message method)
        prop = self.propagate(edge_index, size=(x.size(0), x.size(0)), v=x,
                              pos=pos)

        # append identity channel to the stack
        prop = torch.cat([prop, x], axis=1)

        # mix channels via neural network (our gamma function)
        prop = self.lin(prop)
        return prop

    def message(self, v_i, v_j, pos_i, pos_j):
        # v and pos are automatically split by propagate before coming into
        # message
        # v_i/j has shape [E, in_channels]
        # pos_i/j has shape [N * in_channels, 2]

        # get relative positions
        diff = (pos_i - pos_j).t()
        r2 = (diff ** 2).sum(axis=0)

        # use r_correction to avoid changing scale (r_{x|y} / r^2)
        diff_x, diff_y = diff * self.r_correction / (r2.view(1, -1) + 0.01)

        # deleting temp variables to free memory
        del diff
        del r2

        diff_x = diff_x.view(-1,1)
        diff_y = diff_y.view(-1,1)

        # this is the x_i - x_j from equation 3
        v_diff = v_i - v_j

        f = torch.cat([v_diff * diff_x, v_diff * diff_y, v_j], axis=1)
        return f

    def update(self, aggr_out):
        return aggr_out


class DPOConv3D(DPOConv):
    '''
    Performs 3D convolution using PDO method.

    Parameters:
        in_channels: number of input channels
        out_channels: number of output channels
        r_correction: factor to adapt r by (should be such that the r value is
                      around 1)

        returns nodes with out_channels channels for each
    '''
    def __init__(self, in_channels, out_channels, r_correction=28):
        super(DPOConv3D, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         r_correction=r_correction)

        # every input channel has 4 items calculated for it before going
        # through neural network
        self.lin = nn.Linear(5 * in_channels, out_channels)

    def message(self, v_i, v_j, pos_i, pos_j):
        # v and pos are automatically split by propagate before coming into
        # message
        # v_i/j has shape [E, in_channels]
        # pos_i/j has shape [N * in_channels, 2]

        # get relative positions
        diff = (pos_i - pos_j).t()
        r2 = (diff ** 2).sum(axis=0)

        # use r_correction to avoid changing scale (r_{x|y} / r^2)
        diff_x, diff_y, diff_z = diff * self.r_correction / (r2.view(1, -1) + 0.01)

        # deleting temp variables to free memory
        del diff
        del r2

        diff_x = diff_x.view(-1,1)
        diff_y = diff_y.view(-1,1)
        diff_z = diff_z.view(-1,1)

        # this is the x_i - x_j from equation 3
        v_diff = v_i - v_j

        f = torch.cat([v_diff * diff_x, v_diff * diff_y, v_diff * diff_z, v_j], axis=1)

        return f


class DownsampleModule(nn.Module):
    '''
    Performs convolution and downsampling.

    Parameters:
        initial_features: number of input channels
        output_features: number of output channels
        voxel_size: size parameter for voxel_grid call
        grid_scale: tuple of minimum position value and largest position value
            defaults to (0, 1)

        returns the convolved and pooled output, input node positions, and input batch for
        use in upsampling/skip connections
    '''
    def __init__(self, initial_features, output_features, voxel_size,
                 grid_scale=(0,1), dimensions=2, mixing_layer=DPOConv):
        super(DownsampleModule, self).__init__()
        r_correction = max(grid_scale) - min(grid_scale)
        self.conv = mixing_layer(initial_features, output_features, r_correction)
        self.grid_scale = grid_scale
        self.voxel_size = voxel_size
        grid_size = self.grid_scale[1] - self.grid_scale[0]
        self.voxels_out = int(grid_size / self.voxel_size)
        self.output_size = self.voxels_out ** dimensions * output_features

    def __str__(self):
        return "grid_scale: {}, voxel_size: {}, voxels_out: {}\nconv: {}\n".format(
            self.grid_scale, self.voxel_size, self.voxels_out,
            self.conv.__str__()
        )

    def forward(self, data):
        orig_pos, orig_batch = data.pos, data.batch
        data.x = F.relu(self.conv(data.x, data.edge_index, data.pos))
        cluster = voxel_grid(data.pos, data.batch, size=self.voxel_size,
                             start=self.grid_scale[0], end=self.grid_scale[1])
        data.edge_attr = None
        data = max_pool(cluster, data)
        return data, orig_pos, orig_batch


class UpsampleModule(nn.Module):
    '''
    Performs convolution and upsampling.

    Parameters:
        initial_features: number of input channels
        output_features: number of output channels
        k: pass to knn_interpolate for upsampling
        mixing_layer: layer to use for mixing prior to upsample

        returns the convolved and unsampled output
    '''
    def __init__(self, initial_features, output_features, k,
                 mixing_layer=DPOConv):
        super(UpsampleModule, self).__init__()
        self.k = k
        self.conv = mixing_layer(initial_features, output_features, 1)
        self.lin1 = nn.Linear(output_features, output_features)
        self.lin2 = nn.Linear(output_features, output_features)
        self.lin3 = nn.Linear(output_features, output_features)
        self.lin4 = nn.Linear(output_features, output_features)

    def __str__(self):
        return self.conv.__str__()

    def forward(self, data, pos_skip, batch_skip, x_skip=None):
        # mix
        data.x = self.conv(data.x, data.edge_index, data.pos)
        data.x = F.elu(data.x)
        data.x = self.lin2(data.x)
        data.x = F.elu(data.x)

        # upsample
        x, pos, batch = data.x, data.pos, data.batch
        data.x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        data.pos, data.batch = pos_skip, batch_skip
        data = triangulate(data)
        if x_skip:
            data.x = torch.cat([x, x_skip], dim=1)
        data.pos, data.batch = pos_skip, batch_skip

        return data
