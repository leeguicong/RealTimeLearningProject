from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class KernelContext:
    sensor_tensor: List[float]
    dt: float
    dt_scale: float
    update_stride: int
    active_region_mask: List[Any]
    trace_only_mask: bool
    precision_target: float
    mode: str


class RuntimeKernelBridge:
    """Bridge runtime outputs to kernel-consumable execution context."""

    def build_kernel_context(
        self,
        evidence,
        runtime_result,
        energy_gates,
        default_dt: float,
        lnc_registry,
        lnc_table,
        lsb_tensor,
    ) -> Dict[str, Any]:
        scheduler_output = getattr(runtime_result, "scheduler_output", {}) if runtime_result is not None else {}
        mode = str(scheduler_output.get("mode", "idle_drift"))
        suspension_action = str(scheduler_output.get("suspension_action", "UPDATE"))

        dt_mul = {
            "active_burst": 0.5,
            "idle_drift": 1.0,
            "sleep": 2.0,
        }.get(mode, 1.0)
        dt = float(default_dt) * dt_mul

        sensor_tensor = [float(x) for x in getattr(evidence, "e_val", [])]
        quality = float(getattr(evidence, "quality", 1.0))
        precision_target = min(quality, 0.4 if not energy_gates.get("allow_high_precision_obs", True) else 1.0)

        selected_ids = scheduler_output.get("selected_lncs", []) if isinstance(scheduler_output, dict) else []
        active_region_mask = []
        for lid in selected_ids:
            state = (lnc_registry or {}).get(lid, {})
            region = state.get("region")
            if region is not None:
                active_region_mask.append(region)
            else:
                active_region_mask.append(lid)

        context = KernelContext(
            sensor_tensor=sensor_tensor,
            dt=dt,
            dt_scale=float(energy_gates.get("max_dt_scale", 1.0)),
            update_stride=int(energy_gates.get("min_update_stride", 1)),
            active_region_mask=active_region_mask,
            trace_only_mask=suspension_action in ("TRACE_ONLY", "SLEEP"),
            precision_target=precision_target,
            mode=mode,
        )
        return context.__dict__
