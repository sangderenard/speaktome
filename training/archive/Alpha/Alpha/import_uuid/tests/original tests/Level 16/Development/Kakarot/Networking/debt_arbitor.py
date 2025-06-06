# debt_arbitor.py

import torch
import logging
from .helper_functions import conditional_round, interpret_exchange_types

class DebtArbitor:
    def __init__(self):
        self.correction_logs = []
        self.fault_history = []

    def arbitrate(self, error, bounds, force_limits, exchange_rates, current_debt, debt_threshold, exchange_types):
        exchange_types = interpret_exchange_types(exchange_types)
        correction_suggestion = error * bounds
        scaled_correction = torch.min(torch.abs(correction_suggestion), force_limits) * torch.sign(correction_suggestion)

        # Determine exchange types per edge
        final_corrections = []
        updated_debts = []
        for i in range(scaled_correction.size(0)):
            exchange_type = "float" if exchange_types[i][0] == exchange_types[i][1] == "float" else "integer"
            correction_acceptance = conditional_round(scaled_correction[i], exchange_type)
            energy_cost = torch.sum(torch.abs(correction_acceptance) * exchange_rates[i])

            if current_debt[i] + energy_cost > debt_threshold[i]:
                correction_acceptance = torch.zeros_like(correction_acceptance)
            else:
                current_debt[i] += energy_cost

            final_corrections.append(correction_acceptance)
            updated_debts.append(current_debt[i])

        final_corrections = torch.stack(final_corrections)
        updated_debt = torch.tensor(updated_debts)

        self.correction_logs.append({
            "error": error.clone(),
            "suggestion": correction_suggestion.clone(),
            "acceptance": final_corrections.clone(),
            "energy_cost": energy_cost.clone(),
            "updated_debt": updated_debt.clone(),
        })

        logging.info(f"Debt arbitration performed. Correction acceptance: {final_corrections}, Updated debt: {updated_debt}")

        return final_corrections, updated_debt
