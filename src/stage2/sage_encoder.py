import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, out_channels=64, dropout=0.2):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, out_channels)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):

        h1 = self.dropout(self.act(self.conv1(x, edge_index)))

        h2 = self.dropout(self.act(self.conv2(h1, edge_index))) + h1

        h3 = self.dropout(self.act(self.conv3(h2, edge_index))) + h2

        out = self.conv4(h3, edge_index)

        return F.normalize(out, p=2, dim=1)