from torch import nn
import torch.nn.functional as F
from models.model_utils import DownsampleModule


class Net(nn.Module):
    '''
        Creates a superpixel classification model using the DPO layer

        Parameters:
            input_channels: number of channels on input
            output_channels: number of classes for output
            depth: how many downsamplings passes to go through
            dropout_rate: dropout rate before the fully connected layers
            grid_scale: describes the size of the input space
            initial_features: how many features does the model start with after
                              first layer
            max_features: upper limit for feature count
            voxels_at_depth: how many n x n voxels are present before fully
                             connected layers

            returns raw outputs from final fully connected layer
    '''
    def __init__(self, input_channels, output_channels, depth=4,
                 dropout_rate=0.4, grid_scale=(0, 1), initial_features=16,
                 voxels_at_depth=2, max_features=256):
        super(Net, self).__init__()
        self.dropout_rate = dropout_rate
        grid_size = grid_scale[1] - grid_scale[0]
        self.downsamples = nn.ModuleList()
        self.upsample = nn.ModuleList()
        features = min(initial_features, max_features)

        self.downsamples.append(
            DownsampleModule(
                input_channels, features,
                grid_size / (voxels_at_depth * 2 ** (depth - 1)),
                grid_scale, dimensions=2
            )
        )

        # for deeper networks, the voxel pooling on initial layers is probably
        # not actually changing the graph
        for i in range(1, depth):
            if features >= max_features:
                self.downsamples.append(
                    DownsampleModule(
                        features, max_features,
                        grid_size / (voxels_at_depth * 2 ** (depth - 1 - i)),
                        grid_scale, dimensions=2
                    )
                )
                features = max_features
            else:
                self.downsamples.append(
                    DownsampleModule(
                        features, features * 2,
                        grid_size / (voxels_at_depth * 2 ** (depth - 1 - i)),
                        grid_scale, dimensions=2
                    )
                )
                features *= 2

        output_size = self.downsamples[-1].output_size
        self.fc1 = nn.Linear(output_size, output_size // 2)
        self.fc2 = nn.Linear(output_size // 2, output_channels)

    def __str__(self):
        # provides a text version of the model description
        string = ''
        for ds in self.downsamples:
            string += ds.__str__()
        string += self.fc1.__str__() + "\n"
        string += self.fc2.__str__() + "\n"
        return string

    def forward(self, data):
        for ds in self.downsamples:
            data, _, _ = ds(data)

        # flattens out the final voxel grid output before fully connected
        # layers
        x = data.x.view(-1, self.fc1.weight.size(1))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)

        return x
