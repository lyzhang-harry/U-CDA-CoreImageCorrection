import torch
import torch.nn as nn


class ColumnBackbone(nn.Module):
    # simple 1d cnn to grab column context
    def __init__(self, F, C=48):
        super().__init__()
        self.conv1 = nn.Conv1d(F, C, 5, padding=2)
        self.conv2 = nn.Conv1d(C, C, 5, padding=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))))


class Heads(nn.Module):
    # output heads for the three branches: Y, HSV, and Lab
    def __init__(self, C, out_per_head=2):
        super().__init__()
        self.y_head = nn.Conv1d(C, out_per_head, 1)
        self.hs_head = nn.Conv1d(C, out_per_head, 1)
        self.ab_head = nn.Conv1d(C, out_per_head, 1)

        # init with zeros so it doesn't jump around at the start of training
        for m in [self.y_head, self.hs_head, self.ab_head]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h):
        return self.y_head(h), self.hs_head(h), self.ab_head(h)