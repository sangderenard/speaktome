# contract_renegotiator.py

import torch
import logging

class ContractRenegotiator:
    def __init__(self, renegotiation_threshold=0.8):
        self.renegotiation_threshold = renegotiation_threshold

    def suggest_renegotiation(self, edge_attr):
        current_debt = edge_attr["current_debt"]
        debt_threshold = edge_attr["debt_threshold"]
        bounds = edge_attr["bounds"]

        renegotiation_mask = current_debt / debt_threshold > self.renegotiation_threshold
        logging.info(f"Renegotiation mask calculated: {renegotiation_mask}")

        renegotiated_bounds = bounds.clone()

        # Handle single-value bounds (1D tensor)
        if renegotiated_bounds.dim() == 1:
            if renegotiation_mask.any():
                renegotiated_bounds[renegotiation_mask] = 0
                logging.info(f"Released bound for edges: {torch.where(renegotiation_mask)[0]}")
        # Handle batch bounds (2D tensor)
        elif renegotiated_bounds.dim() == 2:
            for i in range(renegotiated_bounds.size(1)):  # Iterate over feature dimensions
                if renegotiation_mask.any():
                    renegotiated_bounds[renegotiation_mask, i] = 0
                    logging.info(f"Released bound on axis {i} for edges: {torch.where(renegotiation_mask)[0]}")
        else:
            raise ValueError(f"Unsupported bounds dimension: {renegotiated_bounds.dim()}")

        return renegotiated_bounds
