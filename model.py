import torch
import torch.nn as nn
import torch.nn.functional as F

N_HIDDEN = 64*64


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        self.hidden_layer = nn.Linear(7*8*8, N_HIDDEN)
        self.output_layer = nn.Linear(N_HIDDEN, 64*64)

    def forward(self, x):
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
