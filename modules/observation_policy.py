from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ObservationConfig:
    sensors: List[str]
    sampling_rate: float
    window_length: float
    resolution: float
    recheck_pointers: List[str]
    precision_target: float
    obs_budget_cost: float
    mode: str


class ObservationPolicy:
    def propose_observation_policies(self, state: Dict[str, float], diagnostics: Dict[str, float]) -> List[ObservationConfig]:
        residual = max(0.0, float(diagnostics.get("residual_energy", 0.0)))

        low = ObservationConfig(
            sensors=["text"],
            sampling_rate=0.5,
            window_length=1.0,
            resolution=0.4,
            recheck_pointers=["global_context"],
            precision_target=0.35,
            obs_budget_cost=0.12,
            mode="low_res_scan",
        )
        high = ObservationConfig(
            sensors=["text", "action"],
            sampling_rate=1.0,
            window_length=2.0,
            resolution=1.0,
            recheck_pointers=["residual_hotspot", "ghost_peak"],
            precision_target=0.8,
            obs_budget_cost=0.35,
            mode="high_res_recheck",
        )

        if residual > 0.5:
            return [high, low]
        return [low, high]

    def score_policy_G(self,
                       config: ObservationConfig,
                       state: Dict[str, float],
                       diagnostics: Dict[str, float],
                       max_obs_budget: float = 1.0) -> float:
        residual = max(0.0, float(diagnostics.get("residual_energy", 0.0)))
        uncertainty = max(0.0, float(diagnostics.get("uncertainty", 0.2)))
        info_gain = residual * config.precision_target * (0.8 + uncertainty)
        affordability = max(0.0, config.obs_budget_cost / max(1e-6, max_obs_budget))
        return affordability - info_gain

    def select_observation_policy(self,
                                  candidates: List[ObservationConfig],
                                  budget: float,
                                  state: Dict[str, float],
                                  diagnostics: Dict[str, float]) -> ObservationConfig:
        scored = []
        for cfg in candidates:
            if cfg.obs_budget_cost <= budget:
                scored.append((self.score_policy_G(cfg, state, diagnostics, max_obs_budget=budget), cfg))
        if not scored:
            return min(candidates, key=lambda x: x.obs_budget_cost)
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
