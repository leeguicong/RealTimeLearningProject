from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class ActionRouter:
    """Role-based action routing from LNC/LSB to output controllers."""

    def __init__(self, lnc_manager: Any, controllers: Optional[Dict[str, Any]] = None):
        self.lnc_manager = lnc_manager
        self.controllers = controllers or {}

    def register_controller(self, role: str, controller: Any) -> None:
        self.controllers[role] = controller

    def resolve_action_target(self, role: str) -> Dict[str, Any]:
        lnc_id = self.lnc_manager.get_role_bound_lnc(role)
        region = self.lnc_manager.get_role_bound_region(role)
        return {"role": role, "lnc_id": lnc_id, "region": region}

    def extract_action_tensor(self, role: str, lsb_tensor, output_spec: Tuple[Tuple[int, ...], Any]):
        target = self.resolve_action_target(role)
        region = target.get("region")
        shape, _dtype = output_spec
        total = 1
        for dim in shape:
            total *= dim

        flat = []
        if region is not None and lsb_tensor is not None:
            x, y, w, h = region
            try:
                patch = lsb_tensor[y:y + h, x:x + w]
                if hasattr(patch, "reshape"):
                    flat = [float(v) for v in patch.reshape(-1)[:total]]
                else:
                    for row in patch:
                        if isinstance(row, (list, tuple)):
                            for v in row:
                                if isinstance(v, (list, tuple)):
                                    flat.extend(float(x) for x in v)
                                else:
                                    flat.append(float(v))
                        else:
                            flat.append(float(row))
            except Exception:
                flat = []

        if len(flat) < total:
            flat.extend([0.0] * (total - len(flat)))
        return flat[:total]

    async def dispatch(self, role: str, action_tensor) -> Dict[str, Any]:
        controller = self.controllers.get(role)
        if controller is None:
            return {"ok": False, "reason": "missing_controller", "role": role}
        await controller.execute_oscillator_command(action_tensor)
        return {"ok": True, "role": role}

    def can_dispatch(self, role: str, energy_gates: Dict[str, Any], action_cost: float, energy_available: float) -> Dict[str, Any]:
        if role == "motor" and not energy_gates.get("allow_burst", True):
            return {"allow": False, "reason": "gate_blocked_burst"}
        if action_cost > energy_available and not energy_gates.get("allow_overdraft", False):
            return {"allow": False, "reason": "insufficient_energy"}
        return {"allow": True, "reason": "ok"}
