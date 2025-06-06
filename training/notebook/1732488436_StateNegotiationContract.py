# state_negotiation_contract.py

import torch
import logging
from .helper_functions import conditional_round

class StateNegotiationContract:
    def __init__(self, bounds, force_limits, exchange_rates, debt_threshold, exchange_type="float"):
        self.bounds = bounds
        self.force_limits = force_limits
        self.exchange_rates = exchange_rates
        self.debt_threshold = torch.tensor(debt_threshold)
        self.current_debt = torch.zeros_like(self.debt_threshold)
        self.energy_history = []
        self.error_history = []
        self.exchange_type = exchange_type

    def apply_correction(self, correction, error):
        scaled_correction = torch.min(torch.abs(correction), self.force_limits) * torch.sign(correction)
        bounded_correction = scaled_correction * self.bounds
        bounded_correction = conditional_round(bounded_correction, self.exchange_type)
        energy_cost = torch.sum(torch.abs(bounded_correction) * self.exchange_rates)

        self.error_history.append(error.clone().detach())

        if self.current_debt + energy_cost > self.debt_threshold:
            logging.info(
                f"Correction rejected. Correction: {correction}, "
                f"Energy cost: {energy_cost}, Current debt: {self.current_debt}, "
                f"Debt threshold: {self.debt_threshold}"
            )
            return torch.zeros_like(correction)
        else:
            self.current_debt += energy_cost
            self.energy_history.append(self.current_debt.clone().detach())
            logging.info(
                f"Correction accepted. Correction: {correction}, "
                f"Bounded correction: {bounded_correction}, "
                f"Energy cost: {energy_cost}, Current debt: {self.current_debt}"
            )
            return bounded_correction

    def regenerate_energy(self, radiation_absorbed, dt):
        regen_amount = torch.tensor(radiation_absorbed * dt)
        old_debt = self.current_debt.clone()
        self.current_debt = torch.max(torch.tensor(0.0), self.current_debt - regen_amount)
        logging.info(
            f"Energy regenerated. Regen amount: {regen_amount}, "
            f"Old debt: {old_debt}, New debt: {self.current_debt}"
        )
