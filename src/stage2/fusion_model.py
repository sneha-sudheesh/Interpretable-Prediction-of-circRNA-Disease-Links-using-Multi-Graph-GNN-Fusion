import torch
import torch.nn as nn


class FusionPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, circ_emb, dis_emb):

        x = torch.cat([circ_emb, dis_emb], dim=1)

        return self.mlp(x).squeeze()