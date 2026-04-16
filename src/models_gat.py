import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(nn.Module):

    def __init__(self, in_channels, hidden_channels=64,
                 out_channels=64, heads=4, dropout=0.2):
        super().__init__()

        # ===== Relation template =====
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
            {
                rel: GATConv(in_channels, hidden_channels,
                             heads=heads,
                             dropout=dropout,
                             add_self_loops=False)
                for rel in relations
            },
            aggr="mean"
        )


        # -------- Output Layer --------
        self.conv2 = HeteroConv(
            {
                rel: GATConv(hidden_channels * heads,
                             out_channels,
                             heads=1,
                             concat=False,
                             add_self_loops=False)
                for rel in relations
            },
            aggr="mean"
        )

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):

        # Layer 1
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: self.dropout(self.act(v)) for k, v in x.items()}

        # Output
        x = self.conv2(x, edge_index_dict)

        # Normalize for dot-product decoder
        x = {k: F.normalize(v, p=2, dim=1) for k, v in x.items()}

        return x