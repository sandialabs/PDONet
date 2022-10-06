from torch import nn
import torch.nn.functional as F
from models.model_utils import DPOConv3D


class Net(nn.Module):
    def __init__(self, input_features=3, output_channels=6890, depth=8,
                 dropout_rate=0.3, grid_scale=(-1.2432, 1.1599),
                 initial_features=16, max_features=128):
        super(Net, self).__init__()
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        features = initial_features
        r_factor = max(grid_scale) - min(grid_scale)
        self.convs.append(
            DPOConv3D(
                input_features,
                features,
                r_factor
            )
        )
        for _ in range(1, depth):
            if features >= max_features:
                features = max_features
                self.convs.append(
                    DPOConv3D(
                        features,
                        features,
                        r_factor
                    )
                )
            else:
                self.convs.append(
                    DPOConv3D(
                        features,
                        features*2,
                        r_factor
                    )
                )
                features *= 2

        output_size = features
        self.fc1 = nn.Linear(output_size, output_size * 2)
        self.fc2 = nn.Linear(output_size * 2, output_channels)

    def __str__(self):
        string = ''
        for conv in self.convs:
            string += conv.__str__() + "\n"
        string += self.fc1.__str__() + "\n"
        string += self.fc2.__str__() + "\n"
        return string

    def forward(self, data):
        for conv in self.convs:
            data.x = F.selu(conv(data.x, data.edge_index, data.pos))

        x = data.x
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)

        return x
