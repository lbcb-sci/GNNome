import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """
    Module that encodes the node features into a high-dimensional
    vector.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layer used to encode the edge attributes
    """

    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        """
        Parameters:
        in_channels : int
            Dimension of the input vectors
        out_channels : int
            Dimension of the output (encoded) vectors
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Return the encoded node attributes."""
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
