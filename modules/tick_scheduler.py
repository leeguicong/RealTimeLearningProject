from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import math


class RuntimeMode(str, Enum):
    ACTIVE_BURST = "active_burst"
    IDLE_DRIFT = "idle_drift"
    SLEEP = "sleep"


@dataclass
class EvidencePacket:
    """统一输入契约: e_key/e_val/e_rel/e_res_hint + 感官时间戳。"""
    e_key: List[float]
    e_val: List[float]
    e_rel: Dict[str, float]
    e_res_hint: Optional[List[float]] = None
    ts_sense: float = 0.0
    quality: float = 1.0


@dataclass
class SuspensionHypothesis:
    hyp_type: str
    weight: float
    predicted_effect: str
    ttl: int
    budget: float


@dataclass
class ShadowBranch:
    branch_id: str
    parent_lnc: str
    score: float
    ttl: int
    status: str = "trial"


class TickScheduler:
    """认知绑定调度器 (Cognitive Binding Scheduler)。

    对齐文档目标：
    - 三模式运行: burst / drift / sleep
    - 接收 EvidencePacket 驱动调度
    - 维护简单悬置候选（H_time/H_split/H_noise/H_seek）
    """

    def __init__(self, energy_budget_limit: float):
        self.energy_budget_limit = energy_budget_limit
        self.current_mode: RuntimeMode = RuntimeMode.IDLE_DRIFT
        self.suspension_queue: List[SuspensionHypothesis] = []
        self.shadow_branches: Dict[str, ShadowBranch] = {}
        self._shadow_counter: int = 0

    def _estimate_residual(self, evidence: Optional[EvidencePacket], preloader_results: List[Any]) -> float:
        if evidence is None:
            return 0.0
        hint = sum(abs(x) for x in (evidence.e_res_hint or []))
        key_norm = sum(abs(x) for x in evidence.e_key)
        novelty = 0.0 if preloader_results else 0.25
        return min(1.0, 0.05 * hint + 0.01 * key_norm + novelty)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        aa, bb = a[:n], b[:n]
        na = math.sqrt(sum(x * x for x in aa))
        nb = math.sqrt(sum(y * y for y in bb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return sum(x * y for x, y in zip(aa, bb)) / (na * nb)

    def _select_mode(self, evidence: Optional[EvidencePacket], residual: float, nt_state: Dict[str, float]) -> RuntimeMode:
        budget = nt_state.get("energy_budget", self.energy_budget_limit)
        has_new_evidence = evidence is not None and evidence.quality > 0.0

        if budget <= 0.05 * self.energy_budget_limit:
            return RuntimeMode.SLEEP
        if has_new_evidence or residual > 0.4:
            return RuntimeMode.ACTIVE_BURST
        return RuntimeMode.IDLE_DRIFT

    def _build_suspension_hypotheses(self, residual: float, evidence: Optional[EvidencePacket]) -> List[SuspensionHypothesis]:
        if evidence is None:
            return []

        hypotheses: List[SuspensionHypothesis] = []
        dt_anomaly = abs(evidence.e_rel.get("delta_t", 0.0))
        if dt_anomaly > 0.2:
            hypotheses.append(SuspensionHypothesis("H_time", min(1.0, 0.5 + dt_anomaly), "align frame alpha/phi", 8, 0.2))
        if residual > 0.6:
            hypotheses.append(SuspensionHypothesis("H_split", residual, "spawn shadow lnc candidate", 12, 0.4))
        if evidence.quality < 0.3:
            hypotheses.append(SuspensionHypothesis("H_noise", 1.0 - evidence.quality, "freeze update and write traces", 6, 0.1))
        if residual > 0.3 and evidence.quality >= 0.3:
            hypotheses.append(SuspensionHypothesis("H_seek", residual, "expand recall candidate set", 10, 0.3))
        return hypotheses

    def perform_binding(self,
                        lnc_registry: Dict[str, Dict[str, float]],
                        preloader_results: List[str],
                        nt_state: Dict[str, float],
                        evidence_packet: Optional[EvidencePacket] = None,
                        energy_gates: Optional[Dict[str, Any]] = None,
                        efe_score: Optional[float] = None) -> Dict[str, Any]:
        """执行一次绑定并返回调度计划。"""
        residual = self._estimate_residual(evidence_packet, preloader_results)
        self.current_mode = self._select_mode(evidence_packet, residual, nt_state)

        if energy_gates and not energy_gates.get("allow_burst", True) and self.current_mode == RuntimeMode.ACTIVE_BURST:
            self.current_mode = RuntimeMode.IDLE_DRIFT
        if energy_gates and energy_gates.get("mode_bias", {}).get("sleep", 0.0) > 0.5:
            self.current_mode = RuntimeMode.SLEEP

        # Priority = (NT_Precision * Resonance_Score) / (Current_Tension + EPS)
        nt_precision = nt_state.get("precision", 1.0)
        eps = 1e-6

        candidates = preloader_results if preloader_results else list(lnc_registry.keys())
        scored = []
        for lnc_id in candidates:
            state = lnc_registry.get(lnc_id, {})
            resonance_score = state.get("resonance", 0.5)
            tension = state.get("tension", 0.1)

            # 文档对齐：w_i ∝ gate_i * φ(sim(q_key,key_i)) * ψ(budget_i)
            key_sig = state.get("key_sig", [])
            sim = self._cosine_similarity(evidence_packet.e_key, key_sig) if evidence_packet is not None else 0.0
            phi = max(0.0, sim)  # ReLU-like
            gate_i = float(state.get("gate", 1.0))
            budget_i = float(state.get("budget", 1.0))
            psi = max(0.0, budget_i)

            if phi <= 0.0:
                # 向后兼容：若还未提供 key_sig，回退旧共振打分
                base_score = (nt_precision * resonance_score) / (tension + eps)
            else:
                base_score = gate_i * phi * psi

            score = base_score / (tension + eps)
            scored.append((lnc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        mode_top_k = {
            RuntimeMode.ACTIVE_BURST: 8,
            RuntimeMode.IDLE_DRIFT: 3,
            RuntimeMode.SLEEP: 0,
        }[self.current_mode]

        if energy_gates:
            if not energy_gates.get("allow_replay", True):
                mode_top_k = min(mode_top_k, 2)
            if not energy_gates.get("allow_structure_split", True):
                mode_top_k = min(mode_top_k, 4)

        selected = [lnc_id for lnc_id, _ in scored[:mode_top_k]]
        hypotheses = self._build_suspension_hypotheses(residual, evidence_packet)
        self.suspension_queue.extend(hypotheses)

        shadow_events = self._update_shadow_branches(selected, hypotheses, residual)
        structure_plan = self._derive_structure_plan(shadow_events)

        epiphany_triggered = residual > 0.65 and len(preloader_results) > 0
        epiphany_budget = 0.2 if epiphany_triggered else 0.0

        return {
            "mode": self.current_mode.value,
            "selected_lncs": selected,
            "routing_entropy_proxy": float(len(selected) / max(1, len(candidates))),
            "residual_energy_proxy": residual,
            "efe_score": float(efe_score) if efe_score is not None else residual,
            "suspension_hypotheses": [h.__dict__ for h in hypotheses],
            "epiphany_signal": {
                "triggered": epiphany_triggered,
                "target_lncs": selected[:3],
                "budget": epiphany_budget,
                "reason": "ghost_resonance+high_residual" if epiphany_triggered else "none",
            },
            "shadow_plan": {
                "active_branches": [b.__dict__ for b in self.shadow_branches.values()],
                "events": shadow_events,
            },
            "replay_window": {
                "enabled": self.current_mode in (RuntimeMode.IDLE_DRIFT, RuntimeMode.SLEEP) and (energy_gates.get("allow_replay", True) if energy_gates else True),
                "max_items": 2 if self.current_mode == RuntimeMode.IDLE_DRIFT else 4,
            },
            "structure_plan": structure_plan,
        }

    def _update_shadow_branches(self,
                                selected_lncs: List[str],
                                hypotheses: List[SuspensionHypothesis],
                                residual: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        has_split = any(h.hyp_type == "H_split" for h in hypotheses)
        if has_split and selected_lncs:
            parent = selected_lncs[0]
            self._shadow_counter += 1
            bid = f"shadow-{parent}-{self._shadow_counter}"
            self.shadow_branches[bid] = ShadowBranch(
                branch_id=bid,
                parent_lnc=parent,
                score=residual,
                ttl=6,
            )
            events.append({"event": "shadow_branch_spawned", "branch_id": bid, "parent": parent})

        for bid, branch in list(self.shadow_branches.items()):
            branch.ttl -= 1
            branch.score = 0.7 * branch.score + 0.3 * residual
            if branch.score > 0.75 and branch.status == "trial":
                branch.status = "promoted"
                events.append({"event": "shadow_branch_promoted", "branch_id": bid, "parent": branch.parent_lnc})
            if branch.ttl <= 0:
                events.append({"event": "shadow_branch_discarded", "branch_id": bid, "parent": branch.parent_lnc})
                del self.shadow_branches[bid]

        return events

    def _derive_structure_plan(self, shadow_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        split_commits = []
        merge_hints = []
        for e in shadow_events:
            if e.get("event") == "shadow_branch_promoted":
                split_commits.append({"parent": e.get("parent"), "branch_id": e.get("branch_id")})
            if e.get("event") == "shadow_branch_discarded":
                merge_hints.append({"parent": e.get("parent"), "branch_id": e.get("branch_id")})
        return {
            "split_commits": split_commits,
            "merge_hints": merge_hints,
        }
