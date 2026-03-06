from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EFETerms:
    data_fit: float
    complexity: float
    instability: float
    resource: float
    energy_penalty: float = 0.0
    overdraft_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.data_fit
            + self.complexity
            + self.instability
            + self.resource
            + self.energy_penalty
            + self.overdraft_penalty
        )


class EFEProxyScorer:
    def __init__(self,
                 k_data: float = 1.0,
                 k_complexity: float = 0.3,
                 k_instability: float = 0.4,
                 k_resource: float = 0.5,
                 k_energy: float = 0.4,
                 k_debt: float = 0.6):
        self.k_data = k_data
        self.k_complexity = k_complexity
        self.k_instability = k_instability
        self.k_resource = k_resource
        self.k_energy = k_energy
        self.k_debt = k_debt

    def score(self,
              prediction_error: float,
              precision: float,
              complexity: float,
              instability: float,
              resource_cost: float,
              action_cost: float,
              energy_available: float) -> EFETerms:
        data_fit = self.k_data * max(0.0, precision) * max(0.0, prediction_error)
        complexity_term = self.k_complexity * max(0.0, complexity)
        instability_term = self.k_instability * max(0.0, instability)
        resource_term = self.k_resource * max(0.0, resource_cost)

        eps = 1e-6
        energy_penalty = self.k_energy * (max(0.0, action_cost) / (max(0.0, energy_available) + eps))
        overdraft_penalty = self.k_debt * max(0.0, action_cost - max(0.0, energy_available))

        return EFETerms(
            data_fit=data_fit,
            complexity=complexity_term,
            instability=instability_term,
            resource=resource_term,
            energy_penalty=energy_penalty,
            overdraft_penalty=overdraft_penalty,
        )
