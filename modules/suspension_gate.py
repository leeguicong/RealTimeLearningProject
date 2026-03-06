from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class SuspensionAction(str, Enum):
    UPDATE = "UPDATE"
    TRACE_ONLY = "TRACE_ONLY"
    SPAWN_SHADOW = "SPAWN_SHADOW"
    COMMIT_EVENT = "COMMIT_EVENT"
    SLEEP = "SLEEP"


@dataclass
class SuspensionDecision:
    action: SuspensionAction
    reason: str


class SuspensionGate:
    def decide(self,
               efe_score: float,
               mode: str,
               precision: float,
               resonance: Dict[str, float],
               resource: Dict[str, float],
               energy_gates: Dict[str, bool]) -> SuspensionDecision:
        if mode == "sleep" or resource.get("energy", 1.0) < 0.15:
            return SuspensionDecision(SuspensionAction.SLEEP, "low_energy_or_sleep_mode")

        if not energy_gates.get("allow_burst", True) or precision < 0.3:
            return SuspensionDecision(SuspensionAction.TRACE_ONLY, "energy_or_precision_gated")

        if resonance.get("ghost_peak", 0.0) > 0.7 and efe_score < 1.1:
            return SuspensionDecision(SuspensionAction.COMMIT_EVENT, "ghost_resonance_event")

        if resonance.get("curvature_peak", 0.0) > 0.65 and energy_gates.get("allow_structure_split", False):
            return SuspensionDecision(SuspensionAction.SPAWN_SHADOW, "curvature_peak_spawn_shadow")

        if efe_score < 1.5:
            return SuspensionDecision(SuspensionAction.UPDATE, "efe_accept")

        return SuspensionDecision(SuspensionAction.TRACE_ONLY, "efe_too_high")
