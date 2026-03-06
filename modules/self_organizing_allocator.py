import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from . import neuron_schema as schema
from .lnc_schema import LNC_UPDATE_STRIDE_INDEX, LNC_DT_SCALE_INDEX


class SelfOrganizingAllocator:
    """自组织分配器（Phase2）。

    增加点：
    - epiphany 事件通道
    - 学习成本预算化
    - 对高压力区域进行降速，避免全局漂移
    """

    def __init__(self,
                 nucleation_growth_size: int = 8,
                 pressure_weight: float = 0.3,
                 auction_radius: int = 4,
                 learning_budget: float = 1.0):
        self.nucleation_growth_size = nucleation_growth_size
        self.pressure_weight = pressure_weight
        self.auction_radius = auction_radius
        self.learning_budget = learning_budget

    @torch.no_grad()
    def run_allocation_cycle(self,
                             lsb_tensor: torch.Tensor,
                             lnc_table: torch.Tensor,
                             budget_area: int,
                             do_auction: bool = False,
                             allocation_map: Optional[torch.Tensor] = None,
                             epiphany_signal: Optional[Dict[str, Any]] = None,
                             shadow_plan: Optional[Dict[str, Any]] = None,
                             replay_batch: Optional[list] = None,
                             energy_gates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        value_field = self.compute_value_field(lsb_tensor)
        k = max(1, budget_area // max(1, self.nucleation_growth_size ** 2))
        nuclei = self._select_nuclei(value_field, k)

        spent = self._apply_nucleation_growth(lsb_tensor, lnc_table, nuclei, budget=self.learning_budget)
        pressure = self.compute_pressure_field(value_field)
        self.apply_pressure_adjustments(lsb_tensor, lnc_table, pressure)

        epiphany_accelerated = []
        if epiphany_signal and epiphany_signal.get("triggered", False):
            epiphany_accelerated = self.apply_epiphany_boost(
                lnc_table=lnc_table,
                lnc_ids=epiphany_signal.get("target_lncs", []),
                budget=min(self.learning_budget, epiphany_signal.get("budget", 0.2)),
            )
            spent += min(self.learning_budget, epiphany_signal.get("budget", 0.2))

        shadow_accelerated = []
        if shadow_plan:
            active = shadow_plan.get("active_branches", [])
            shadow_targets = []
            for b in active:
                if b.get("status") in ("trial", "promoted"):
                    parent = b.get("parent_lnc")
                    if isinstance(parent, int):
                        shadow_targets.append(parent)
                    elif isinstance(parent, str) and parent.isdigit():
                        shadow_targets.append(int(parent))
            if shadow_targets:
                shadow_accelerated = self.apply_epiphany_boost(
                    lnc_table=lnc_table,
                    lnc_ids=shadow_targets,
                    budget=min(0.2, self.learning_budget - spent),
                )

        new_allocation = None
        if do_auction and allocation_map is not None:
            new_allocation = self.auction_reassign_chunks(allocation_map, value_field, nuclei, self.auction_radius)

        structure_actions = self.propose_structure_actions(lsb_tensor, shadow_plan)

        if energy_gates:
            max_dt = float(energy_gates.get("max_dt_scale", 2.5))
            min_stride = float(energy_gates.get("min_update_stride", 1))
            lnc_table[:, LNC_DT_SCALE_INDEX] = torch.clamp(lnc_table[:, LNC_DT_SCALE_INDEX], max=max_dt)
            lnc_table[:, LNC_UPDATE_STRIDE_INDEX] = torch.clamp(lnc_table[:, LNC_UPDATE_STRIDE_INDEX], min=min_stride)

        return {
            "nuclei_count": int(nuclei.shape[0]),
            "budget_spent": float(min(spent, self.learning_budget)),
            "epiphany_accelerated": epiphany_accelerated,
            "shadow_accelerated": shadow_accelerated,
            "replay_consumed": len(replay_batch or []),
            "structure_actions": structure_actions,
            "allocation_map": new_allocation,
        }

    @torch.no_grad()
    def propose_structure_actions(self,
                                  lsb_tensor: torch.Tensor,
                                  shadow_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Phase4: 提议分裂/合并动作（预算化决策，实际执行由 LNCManager）。"""
        residual = torch.clamp(lsb_tensor[..., schema.RESIDUAL_ENERGY_INDEX], 0.0, 10.0)
        lnc_ids = lsb_tensor[..., schema.LNC_ID_INDEX].long()

        split_candidates = []
        for lid in torch.unique(lnc_ids).tolist():
            if lid < 0:
                continue
            mask = lnc_ids == lid
            if not mask.any():
                continue
            avg_res = float(residual[mask].mean().item())
            if avg_res > 0.6:
                split_candidates.append({"lnc_id": int(lid), "score": avg_res, "reason": "high_residual_multimodal_proxy"})

        # merge 候选：低残差并且存在多个 active shadow 分支时，收敛回主干。
        merge_candidates = []
        active_shadow = (shadow_plan or {}).get("active_branches", [])
        if len(active_shadow) >= 2:
            parent_counts: Dict[str, int] = {}
            for b in active_shadow:
                p = str(b.get("parent_lnc"))
                parent_counts[p] = parent_counts.get(p, 0) + 1
            for parent, cnt in parent_counts.items():
                if cnt >= 2:
                    merge_candidates.append({"winner": parent, "reason": "anchor_competition_settled", "branches": cnt})

        return {
            "split_candidates": split_candidates[:2],
            "merge_candidates": merge_candidates[:2],
        }

    @torch.no_grad()
    def compute_value_field(self, lsb_tensor: torch.Tensor) -> torch.Tensor:
        pe = lsb_tensor[..., schema.PREDICTION_ERROR_INDEX].abs()
        nt = torch.clamp(lsb_tensor[..., schema.NT_AMPLITUDE_INDEX], 0.0, 1.0)
        residual = torch.clamp(lsb_tensor[..., schema.RESIDUAL_ENERGY_INDEX], 0.0, 1.0)
        return pe * nt + 0.25 * residual

    @torch.no_grad()
    def _select_nuclei(self, value_field: torch.Tensor, k: int) -> torch.Tensor:
        _, w = value_field.shape
        flat = value_field.reshape(-1)
        k = min(k, flat.numel())
        topk = torch.topk(flat, k=k)
        ys = topk.indices // w
        xs = topk.indices % w
        return torch.stack([ys, xs], dim=1)

    @torch.no_grad()
    def _apply_nucleation_growth(self,
                                 lsb_tensor: torch.Tensor,
                                 lnc_table: torch.Tensor,
                                 nuclei: torch.Tensor,
                                 budget: float) -> float:
        h, w, _ = lsb_tensor.shape
        g = self.nucleation_growth_size
        region_ids = lsb_tensor[..., schema.LNC_ID_INDEX].long()

        spent = 0.0
        for y, x in nuclei.tolist():
            if spent >= budget:
                break
            y0, y1 = max(0, y - g), min(h, y + g + 1)
            x0, x1 = max(0, x - g), min(w, x + g + 1)
            ids = torch.unique(region_ids[y0:y1, x0:x1])
            for lid in ids.tolist():
                if lid < 0 or lid >= lnc_table.shape[0]:
                    continue
                lnc_table[lid, LNC_DT_SCALE_INDEX] = torch.clamp(lnc_table[lid, LNC_DT_SCALE_INDEX] + 0.1, max=2.0)
                lnc_table[lid, LNC_UPDATE_STRIDE_INDEX] = torch.clamp(lnc_table[lid, LNC_UPDATE_STRIDE_INDEX] - 0.2, min=1.0)
            spent += 0.05
        return spent

    @torch.no_grad()
    def compute_pressure_field(self, value_field: torch.Tensor) -> torch.Tensor:
        x = value_field.unsqueeze(0).unsqueeze(0)
        device = value_field.device
        dtype = value_field.dtype
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=dtype, device=device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=dtype, device=device)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        return self.pressure_weight * (grad_x.abs() + grad_y.abs()).squeeze(0).squeeze(0)

    @torch.no_grad()
    def apply_pressure_adjustments(self,
                                   lsb_tensor: torch.Tensor,
                                   lnc_table: torch.Tensor,
                                   pressure: torch.Tensor) -> None:
        region_ids = lsb_tensor[..., schema.LNC_ID_INDEX].long()
        threshold = pressure.mean() + pressure.std()
        high_mask = pressure > threshold
        involved_ids = torch.unique(region_ids[high_mask])
        for lid in involved_ids.tolist():
            if lid < 0 or lid >= lnc_table.shape[0]:
                continue
            lnc_table[lid, LNC_UPDATE_STRIDE_INDEX] = torch.max(
                lnc_table[lid, LNC_UPDATE_STRIDE_INDEX],
                torch.tensor(2.0, dtype=lnc_table.dtype, device=lnc_table.device),
            )
            lnc_table[lid, LNC_DT_SCALE_INDEX] = torch.min(
                lnc_table[lid, LNC_DT_SCALE_INDEX],
                torch.tensor(0.75, dtype=lnc_table.dtype, device=lnc_table.device),
            )

    @torch.no_grad()
    def apply_epiphany_boost(self,
                             lnc_table: torch.Tensor,
                             lnc_ids: list,
                             budget: float = 0.2) -> list:
        boosted = []
        step = 0.05
        for lid in lnc_ids:
            if budget <= 0:
                break
            if lid < 0 or lid >= lnc_table.shape[0]:
                continue
            lnc_table[lid, LNC_DT_SCALE_INDEX] = torch.clamp(lnc_table[lid, LNC_DT_SCALE_INDEX] + 0.3, max=2.5)
            lnc_table[lid, LNC_UPDATE_STRIDE_INDEX] = torch.clamp(lnc_table[lid, LNC_UPDATE_STRIDE_INDEX] - 0.5, min=1.0)
            boosted.append(int(lid))
            budget -= step
        return boosted

    @torch.no_grad()
    def auction_reassign_chunks(self,
                                allocation_map: torch.Tensor,
                                value_field: torch.Tensor,
                                nuclei: torch.Tensor,
                                radius: int = 4) -> torch.Tensor:
        h, w = allocation_map.shape
        new_map = allocation_map.clone()
        for cy, cx in nuclei.tolist():
            y0 = max(0, cy - radius)
            y1 = min(h, cy + radius + 1)
            x0 = max(0, cx - radius)
            x1 = min(w, cx + radius + 1)
            sub = new_map[y0:y1, x0:x1]
            occupied_ids = torch.unique(sub[sub >= 0])
            if occupied_ids.numel() == 0:
                continue
            utilities = []
            for lid in occupied_ids.tolist():
                mask = sub == lid
                if mask.any():
                    util = value_field[y0:y1, x0:x1][mask].mean()
                    utilities.append((lid, float(util.item())))
            if not utilities:
                continue
            best_lnc = max(utilities, key=lambda t: t[1])[0]
            new_map[y0:y1, x0:x1][sub == -1] = best_lnc
        return new_map
