#!/usr/bin/env python3
"""Transform utilities for FontMapper."""
from __future__ import annotations

# --- END HEADER ---

import random
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch

class AddRandomNoise:
    def __call__(self, img):
        noise = torch.randn_like(img, device=img.device) * 0.1
        return torch.clamp(img + noise, 0, 1)

class RandomGaussianBlur:
    def __call__(self, img):
        sigma = random.uniform(0.5, 1.5)
        max_kernel_size = max(img.shape[-2:]) // 10
        kernel_size = 1 + 2 * random.randint(0, max_kernel_size + 1)
        return TF.gaussian_blur(img, kernel_size=(kernel_size, kernel_size), sigma=sigma)

class DistortionChain:
    def __init__(self, noise_chance: float = 0.5, blur_chance: float = 0.5):
        self.noise_chance = noise_chance
        self.blur_chance = blur_chance
        self.noise_transform = AddRandomNoise()
        self.blur_transform = RandomGaussianBlur()

    def __call__(self, img):
        if random.random() < self.noise_chance:
            img = self.noise_transform(img)
        if random.random() < self.blur_chance:
            img = self.blur_transform(img)
        return img

class ToTensorAndToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, pic):
        tensor = transforms.ToTensor()(pic)
        tensor = tensor.to(torch.float16)
        tensor = tensor.to(self.device)
        return tensor
