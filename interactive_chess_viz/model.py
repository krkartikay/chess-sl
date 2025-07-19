import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessModel(nn.Module):
    def __init__(self, n_blocks=4, n_channels=16, n_hidden=4096, filter_size=3):
        super(ChessModel, self).__init__()

        self.conv_blocks = nn.Sequential()

        for i in range(n_blocks):
            n_channels_in = 7 if i == 0 else n_channels
            conv_layer = nn.Conv2d(n_channels_in, n_channels, filter_size, padding='same')
            bn_layer = nn.BatchNorm2d(n_channels)
            self.conv_blocks.append(conv_layer)
            self.conv_blocks.append(bn_layer)
            self.conv_blocks.append(nn.ReLU())

        if n_blocks == 0:
            n_channels = 7

        self.hidden_layer = nn.Linear(n_channels*8*8, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 64*64)

    def forward(self, x):
        # Apply conv blocks
        x = self.conv_blocks(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Apply hidden layer
        x = self.hidden_layer(x)
        x = F.relu(x)
        # Output layer
        x = self.output_layer(x)
        x = F.sigmoid(x)
        return x

    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    model = ChessModel()
    print(model)