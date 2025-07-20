import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessModel(nn.Module):
    def __init__(self, n_blocks=4, n_channels=16, n_hidden=4096, filter_size=3):
        super(ChessModel, self).__init__()

        self.conv_blocks = nn.Sequential()

        for i in range(n_blocks):
            n_channels_in = 7 if i == 0 else n_channels
            conv_layer = nn.Conv2d(n_channels_in, n_channels, filter_size, padding='same')  # [batch_size, n_channels_in, 8, 8] -> [batch_size, n_channels, 8, 8]
            bn_layer = nn.BatchNorm2d(n_channels)  # [batch_size, n_channels, 8, 8] -> [batch_size, n_channels, 8, 8]
            self.conv_blocks.append(conv_layer)
            self.conv_blocks.append(bn_layer)
            self.conv_blocks.append(nn.ReLU())  # [batch_size, n_channels, 8, 8] -> [batch_size, n_channels, 8, 8]

        if n_blocks == 0:
            n_channels = 7

        self.hidden_layer = nn.Linear(n_channels*8*8, n_hidden)  # [batch_size, n_channels*64] -> [batch_size, n_hidden]
        self.output_layer = nn.Linear(n_hidden, 64*64)  # [batch_size, n_hidden] -> [batch_size, 64*64]

    def forward(self, x):
        # x: [batch_size, 7, 8, 8] float32
        x = self.conv_blocks(x)  # -> [batch_size, n_channels, 8, 8] float32
        x = x.view(x.size(0), -1)  # -> [batch_size, n_channels*64] float32
        x = self.hidden_layer(x)  # -> [batch_size, n_hidden] float32
        x = F.relu(x)
        x = self.output_layer(x)  # -> [batch_size, 64*64] float32
        x = F.sigmoid(x)
        return x

    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    model = ChessModel()
    print(model)