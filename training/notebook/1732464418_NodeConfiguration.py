# node_configuration.py

import torch

class NodeConfiguration:
    def __init__(self, bounds, force_limits, exchange_rates, debt_threshold, exchange_type="float"):
        """
        Configuration for a node's simulation capacity.
        """
        self.bounds = bounds
        self.force_limits = force_limits
        self.exchange_rates = exchange_rates
        self.debt_threshold = torch.tensor(debt_threshold)
        self.exchange_type = exchange_type

    def __repr__(self):
        return (f"NodeConfiguration(bounds={self.bounds}, force_limits={self.force_limits}, "
                f"exchange_rates={self.exchange_rates}, debt_threshold={self.debt_threshold}, "
                f"exchange_type={self.exchange_type})")
