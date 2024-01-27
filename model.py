import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels_in = 7
        n_channels = N_CHANNELS.get()
        filter_size = FILTER_SIZE.get()
        bn_layer = nn.BatchNorm2d(n_channels_in)
        conv_layer_up = nn.Conv2d(n_channels_in, n_channels, filter_size, padding='same')
        conv_layer_down = nn.Conv2d(n_channels, n_channels_in, filter_size, padding='same')
        self.seq = nn.Sequential(
            bn_layer,
            conv_layer_up,
            nn.ReLU(),
            conv_layer_down,
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        self.conv_blocks = nn.ModuleList()
        for i in range(N_BLOCKS.get()):
            block = ConvBlock()
            self.conv_blocks.append(block)

        self.hidden_layer = nn.Linear(7*8*8, N_HIDDEN.get())
        self.output_layer = nn.Linear(N_HIDDEN.get(), 64*64)

    def forward(self, x):
        # Apply conv blocks
        for block in self.conv_blocks:
            x = x + block(x)
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
