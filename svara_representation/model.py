import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .inception import Encoder

class Model(nn.Module):
    def __init__(self, embed_dim, num_classes, depth, num_features=2, dropout=0.2):
        super().__init__()
        self.dir = os.path.join("checkpoints")

        self.encoder = Encoder(embed_dim, depth, num_features)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        embedding = self.encoder(input)
        embedding = self.gap(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(-1)

    def save(self, filename):
        os.makedirs(self.dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.dir, filename))

    def load(self, filename, device):
        self.load_state_dict(torch.load(os.path.join(self.dir, filename), map_location=device))

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
