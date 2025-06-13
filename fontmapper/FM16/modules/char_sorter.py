#!/usr/bin/env python3
"""Neural network used for character sorting."""
from __future__ import annotations

# --- END HEADER ---

from torch import nn
import torch

class CharSorter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = len(config.charset)
        self.conv1 = nn.Conv2d(1, config.conv1_out, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(config.conv1_out, config.conv2_out, kernel_size=5, stride=1, padding=2)
        dummy_input = torch.autograd.Variable(torch.ones(1, 1, config.height, config.width))
        output = self.conv2(self.pool(self.conv1(dummy_input)))
        output = self.pool(output)
        n_size = output.numel() // output.shape[0]
        self.fc1 = nn.Linear(n_size, config.linear_out)
        self.fc2 = nn.Linear(config.linear_out, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.dropout)
        self.charset = config.charset
        self.charBitmasks = getattr(config, "charBitmasks", None)
        self.demo_width = config.width
        self.demo_height = config.height

    def forward(self, x):
        x = x.view(-1, 1, self.config.height, self.config.width)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
