import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=9, padding=4)
        self.branch2 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=19, padding=9)
        self.branch3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=39, padding=19)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1) + self.residual(x)


class Encoder(nn.Module):
    def __init__(self, embed_dim, depth, num_features):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
                InceptionModule(num_features, embed_dim * 2 ** (depth - 1)),
                nn.ReLU(),
                nn.BatchNorm1d(embed_dim * 2 ** (depth - 1)),
                nn.AvgPool1d(kernel_size=2, stride=2),
            )
        )
        for i in range(depth - 1, 1, -1):
            self.blocks.append(
                nn.Sequential(
                    InceptionModule(embed_dim * 2 ** i, embed_dim * 2 ** (i - 1)),
                    nn.ReLU(),
                    nn.BatchNorm1d(embed_dim * 2 ** (i - 1)),
                    nn.AvgPool1d(kernel_size=2, stride=2),
                )
            )
        self.blocks.append(InceptionModule(embed_dim * 2, embed_dim))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
