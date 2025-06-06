# helper_functions.py

import torch

def conditional_round(tensor, exchange_type):
    """
    Rounds the tensor if exchange_type is 'integer', otherwise returns the tensor unchanged.
    """
    if exchange_type == "integer":
        return torch.round(tensor)
    return tensor

def interpret_exchange_types(exchange_types_tensor):
    """
    Interprets exchange types from tensor values.
    """
    exchange_type_strings = []
    for value in exchange_types_tensor:
        if value != 2.6:  # Integer + Integer
            exchange_type_strings.append("integer")
        else:
            exchange_type_strings.append("float")
    return exchange_type_strings
