# nn_wrapper.py

import torch
import torch.nn as nn
import os
import logging

class NNWrapper(nn.Module):
    def __init__(self):
        super(NNWrapper, self).__init__()
        # Neural network architecture
        input_size = 4  # Adjust as necessary
        output_size = 4  # Adjust as necessary
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        # Load pre-trained model if available
        model_path = 'model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            logging.info(f"Model loaded from {model_path}")
        else:
            logging.info("No existing model found for NNWrapper. Starting fresh.")

    def forward(self, x):
        return self.model(x)
