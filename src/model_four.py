import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGraphSAGE(nn.Module):
    """
    4-layer heterogeneous GraphSAGE
    Input → Hidden1 → Hidden2 → Hidden3 → Output
    NO residual connections
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=64, dropout=0.2):
        super().__init__()

        # -------- Layer 1: Input → Hidden --------
        relations_1 = {
            ("circRNA", "interacts", "miRNA"): SAGEConv((-1, -1), hidden_channels),
            ("miRNA", "interacts", "disease"): SAGEConv((-1, -1), hidden_channels),
            ("circRNA", "associated", "disease"): SAGEConv((-1, -1), hidden_channels),

            ("miRNA", "rev_interacts", "circRNA"): SAGEConv(in_channels, hidden_channels),
            ("disease", "rev_interacts", "miRNA"): SAGEConv(in_channels, hidden_channels),
            ("disease", "rev_associated", "circRNA"): SAGEConv(in_channels, hidden_channels),
        }

        # -------- Layer 2: Hidden → Hidden --------
        relations_2 = {
            ("circRNA", "interacts", "miRNA"): SAGEConv(hidden_channels, hidden_channels),
            ("miRNA", "interacts", "disease"): SAGEConv(hidden_channels, hidden_channels),
            ("circRNA", "associated", "disease"): SAGEConv(hidden_channels, hidden_channels),

            ("miRNA", "rev_interacts", "circRNA"): SAGEConv(hidden_channels, hidden_channels),
            ("disease", "rev_interacts", "miRNA"): SAGEConv(hidden_channels, hidden_channels),
            ("disease", "rev_associated", "circRNA"): SAGEConv(hidden_channels, hidden_channels),
        }

        # -------- Layer 3: Hidden → Hidden --------
        relations_3 = {
            ("circRNA", "interacts", "miRNA"): SAGEConv(hidden_channels, hidden_channels),
            ("miRNA", "interacts", "disease"): SAGEConv(hidden_channels, hidden_channels),
            ("circRNA", "associated", "disease"): SAGEConv(hidden_channels, hidden_channels),

            ("miRNA", "rev_interacts", "circRNA"): SAGEConv(hidden_channels, hidden_channels),
            ("disease", "rev_interacts", "miRNA"): SAGEConv(hidden_channels, hidden_channels),
            ("disease", "rev_associated", "circRNA"): SAGEConv(hidden_channels, hidden_channels),
        }

        # -------- Layer 4: Hidden → Output --------
        relations_4 = {
            ("circRNA", "interacts", "miRNA"): SAGEConv(hidden_channels, out_channels),
            ("miRNA", "interacts", "disease"): SAGEConv(hidden_channels, out_channels),
            ("circRNA", "associated", "disease"): SAGEConv(hidden_channels, out_channels),

            ("miRNA", "rev_interacts", "circRNA"): SAGEConv(hidden_channels, out_channels),
            ("disease", "rev_interacts", "miRNA"): SAGEConv(hidden_channels, out_channels),
            ("disease", "rev_associated", "circRNA"): SAGEConv(hidden_channels, out_channels),
        }

        self.conv1 = HeteroConv(relations_1, aggr="mean")
        self.conv2 = HeteroConv(relations_2, aggr="mean")
        self.conv3 = HeteroConv(relations_3, aggr="mean")
        self.conv4 = HeteroConv(relations_4, aggr="mean")

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        print("4Layer")
        h1 = self.conv1(x_dict, edge_index_dict)
        for k in h1:
            h1[k] = self.dropout(self.act(h1[k]))

        h2 = self.conv2(h1, edge_index_dict)
        for k in h2:
            h2[k] = self.dropout(self.act(h2[k]))

        h3 = self.conv3(h2, edge_index_dict)
        for k in h3:
            h3[k] = self.dropout(self.act(h3[k]))

        out = self.conv4(h3, edge_index_dict)

        out_c = nn.functional.normalize(out["circRNA"], p=2, dim=1)
        out_m = nn.functional.normalize(out["miRNA"], p=2, dim=1)
        out_d = nn.functional.normalize(out["disease"], p=2, dim=1)

        return {
            "circRNA": out_c,
            "miRNA": out_m,
            "disease": out_d
        }
