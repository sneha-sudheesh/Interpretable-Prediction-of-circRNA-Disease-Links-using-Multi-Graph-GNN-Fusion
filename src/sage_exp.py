import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv



class HeteroGraphSAGE(nn.Module):
    """
    3-layer heterogeneous GraphSAGE
    Input → Hidden1 → Hidden2 → Output
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

        # -------- Layer 3: Hidden → Output --------
        relations_3 = {
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

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connections (input → output)
        self.res_lin_circ = nn.Linear(in_channels, out_channels)
        self.res_lin_mir  = nn.Linear(in_channels, out_channels)
        self.res_lin_dis  = nn.Linear(in_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        print("3Layer")
        # -------- Hidden layer 1 --------
        h1 = self.conv1(x_dict, edge_index_dict)
        for k in h1:
            h1[k] = self.dropout(self.act(h1[k]))

        # -------- Hidden layer 2 --------
        h2 = self.conv2(h1, edge_index_dict)
        for k in h2:
            h2[k] = self.dropout(self.act(h2[k]))

        # -------- Output layer --------
        out = self.conv3(h2, edge_index_dict)

        # Residuals
        out_c = out["circRNA"]
        out_m = out["miRNA"] 
        out_d = out["disease"]

        # L2 normalization (good choice for dot-product decoder)
        out_c = nn.functional.normalize(out_c, p=2, dim=1)
        out_m = nn.functional.normalize(out_m, p=2, dim=1)
        out_d = nn.functional.normalize(out_d, p=2, dim=1)

        return {
            "circRNA": out_c,
            "miRNA": out_m,
            "disease": out_d
        }
