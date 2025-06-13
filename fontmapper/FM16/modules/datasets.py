#!/usr/bin/env python3
"""Dataset helpers for FontMapper."""
from __future__ import annotations

# --- END HEADER ---

from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple

class CustomDataset(Dataset):
    def __init__(self, char_bitmasks: List, labels, transform=None):
        self.char_bitmasks = [Image.fromarray(obj).convert("L") for obj in char_bitmasks]
        self.labels = labels.clone()
        self.transform = transform

    def __getitem__(self, idx):
        image = self.char_bitmasks[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label.clone()

    def __len__(self):
        return len(self.char_bitmasks)

class CustomInputDataset(Dataset):
    def __init__(self, tensor_images, network_size: Tuple[int, int], mode: str = "auto", transform=None):
        self.network_size = network_size
        self.tensor_images = tensor_images
        self.transform = transform
        self.mode = mode
        self.prepared_data = self.prepare_data(tensor_images, network_size, mode)

    def prepare_data(self, tensor_images, network_size, mode):
        # Placeholder simplified implementation
        prepared_data = []
        for image in tensor_images:
            prepared_data.append({"sub_image": image})
        return prepared_data

    def __getitem__(self, index):
        item = self.prepared_data[index]
        if self.transform:
            transformed_image = self.transform(item["sub_image"])
            return {**item, "sub_image": transformed_image}
        return item

    def __len__(self):
        return len(self.prepared_data)
