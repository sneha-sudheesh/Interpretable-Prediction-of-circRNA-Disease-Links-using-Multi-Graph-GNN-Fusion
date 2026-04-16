import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphSAGE(nn.Module):
    """
    4-layer heterogeneous GraphSAGE + GIP
    Input → H1 → H2 → H3 → Output
    """

    def __init__(self, in_channels, hidden_channels=64,
                 out_channels=64, dropout=0.2):
        super().__init__()

        # ----- All relations including GIP -----
        relations = [
            ("circRNA", "interacts", "miRNA"),
            ("miRNA", "interacts", "disease"),
            ("circRNA", "associated", "disease"),
            ("miRNA", "rev_interacts", "circRNA"),
            ("disease", "rev_interacts", "miRNA"),
            ("disease", "rev_associated", "circRNA"),

            # GIP relations
            ("circRNA", "gip_sim", "circRNA"),
            ("disease", "gip_sim", "disease"),
        ]

        # -------- Layer 1 --------
        self.conv1 = HeteroConv(
            {rel: SAGEConv((-1, -1), hidden_channels) for rel in relations},
            aggr="mean"
        )

        # -------- Layer 2 --------
        self.conv2 = HeteroConv(
            {rel: SAGEConv(hidden_channels, hidden_channels) for rel in relations},
            aggr="mean"
        )

        # -------- Layer 4 (Output) --------
        self.conv3 = HeteroConv(
            {rel: SAGEConv(hidden_channels, out_channels) for rel in relations},
            aggr="mean"
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):

        # Layer 1
        h1 = self.conv1(x_dict, edge_index_dict)
        h1 = {k: self.dropout(self.act(v)) for k, v in h1.items()}

        # Layer 2
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: self.dropout(self.act(v)) for k, v in h2.items()}

        # Output
        out = self.conv3(h2, edge_index_dict)

        # Normalize embeddings
        out = {k: F.normalize(v, p=2, dim=1) for k, v in out.items()}

        return out