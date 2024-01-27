import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        n_channels = N_CHANNELS.get()
        self.conv_blocks = nn.Sequential()

        for i in range(N_BLOCKS.get()):
            n_channels_in  = 7 if i == 0 else N_CHANNELS.get()
            filter_size = FILTER_SIZE.get()
            conv_layer = nn.Conv2d(n_channels_in, n_channels, filter_size, padding='same')
            bn_layer = nn.BatchNorm2d(n_channels)
            self.conv_blocks.append(conv_layer)
            self.conv_blocks.append(bn_layer)
            self.conv_blocks.append(nn.ReLU())

        if N_BLOCKS.get() == 0:
            n_channels = 7

        self.hidden_layer = nn.Linear(n_channels*8*8, N_HIDDEN.get())
        self.output_layer = nn.Linear(N_HIDDEN.get(), 64*64)

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
