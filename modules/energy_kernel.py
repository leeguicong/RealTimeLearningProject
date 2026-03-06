from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EnergyState:
    E: float
    E_ceiling: float = 1.0
    E_floor: float = 0.0
    basal_metabolic_rate: float = 0.02
    intake_rate: float = 0.01
    stress: float = 0.0
    arousal: float = 0.5
    last_update_ts: float = 0.0


@dataclass
class ActionProfile:
    obs_cost: float = 0.0
    inference_cost: float = 0.0
    plasticity_cost: float = 0.0
    memory_cost: float = 0.0
    structure_cost: float = 0.0
    transport_cost: float = 0.0


@dataclass
class EnergyGates:
    allow_burst: bool
    allow_high_precision_obs: bool
    allow_structure_split: bool
    allow_snapshot: bool
    allow_replay: bool
    max_dt_scale: float
    min_update_stride: int
    max_obs_budget: float
    mode_bias: Dict[str, float]
    allow_overdraft: bool = False


class EnergyKernel:
    def __init__(self,
                 weights: Dict[str, float] | None = None,
                 E_low: float = 0.2,
                 E_split_gate: float = 0.35,
                 E_obs_gate: float = 0.25,
                 k_debt: float = 0.5,
                 recovery_gain: float = 0.05):
        self.weights = weights or {
            "w_obs": 1.0,
            "w_inf": 1.0,
            "w_plast": 1.2,
            "w_mem": 0.8,
            "w_struct": 1.5,
            "w_transport": 0.4,
        }
        self.E_low = E_low
        self.E_split_gate = E_split_gate
        self.E_obs_gate = E_obs_gate
        self.k_debt = k_debt
        self.recovery_gain = recovery_gain

    def estimate_action_cost(self, action_profile: ActionProfile) -> float:
        return (
            self.weights["w_obs"] * action_profile.obs_cost
            + self.weights["w_inf"] * action_profile.inference_cost
            + self.weights["w_plast"] * action_profile.plasticity_cost
            + self.weights["w_mem"] * action_profile.memory_cost
            + self.weights["w_struct"] * action_profile.structure_cost
            + self.weights["w_transport"] * action_profile.transport_cost
        )

    def apply_energy_dynamics(self,
                              energy_state: EnergyState,
                              action_cost: float,
                              diagnostics: Dict[str, float] | None = None) -> EnergyState:
        diagnostics = diagnostics or {}
        residual_stress = max(0.0, float(diagnostics.get("residual_energy", 0.0)) - 0.5) * 0.1

        energy_before = energy_state.E
        overdraft = max(0.0, action_cost - max(0.0, energy_before - energy_state.E_floor))

        E = energy_before + energy_state.intake_rate - energy_state.basal_metabolic_rate - action_cost
        E = min(energy_state.E_ceiling, max(energy_state.E_floor, E))

        stress = max(0.0, energy_state.stress + self.k_debt * overdraft + residual_stress - self.recovery_gain)

        arousal = (
            0.45 * E
            + 0.30 * float(diagnostics.get("residual_energy", 0.0))
            + 0.20 * float(diagnostics.get("precision", 1.0))
            - 0.35 * stress
        )
        arousal = min(1.0, max(0.0, arousal))

        return EnergyState(
            E=E,
            E_ceiling=energy_state.E_ceiling,
            E_floor=energy_state.E_floor,
            basal_metabolic_rate=energy_state.basal_metabolic_rate,
            intake_rate=energy_state.intake_rate,
            stress=stress,
            arousal=arousal,
            last_update_ts=float(diagnostics.get("ts", energy_state.last_update_ts)),
        )

    def energy_gating(self, energy_state: EnergyState, precision: float) -> EnergyGates:
        low = energy_state.E < self.E_low
        return EnergyGates(
            allow_burst=(not low) and energy_state.arousal > 0.3,
            allow_high_precision_obs=energy_state.E >= self.E_obs_gate and precision >= 0.3,
            allow_structure_split=energy_state.E >= self.E_split_gate,
            allow_snapshot=energy_state.E >= self.E_split_gate,
            allow_replay=energy_state.E >= self.E_low,
            max_dt_scale=1.6 if energy_state.E >= self.E_split_gate else 0.9,
            min_update_stride=1 if energy_state.E >= self.E_split_gate else 3,
            max_obs_budget=max(0.05, min(1.0, energy_state.E)),
            mode_bias={
                "sleep": 0.6 if low else 0.1,
                "drift": 0.3 if low else 0.4,
                "burst": 0.1 if low else 0.5,
            },
            allow_overdraft=(precision > 0.6 and energy_state.stress < 0.6),
        )
