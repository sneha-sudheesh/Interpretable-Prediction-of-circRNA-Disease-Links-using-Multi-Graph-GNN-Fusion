import torch
import torch.nn as nn


class WeightedFusionPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, circ1, circ2, dis1, dis2):

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        circ = alpha * circ1 + (1 - alpha) * circ2
        dis = beta * dis1 + (1 - beta) * dis2

        x = torch.cat([circ, dis], dim=1)

        return self.mlp(x).squeeze()