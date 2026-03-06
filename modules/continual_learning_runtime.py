"""Continual learning runtime loop (explicit learning-closure).

将文档中的概念闭环落地为可执行流程：
Routing -> Residual -> Ghost -> ObservationPolicy -> EFE -> Suspension -> Epiphany -> Structure Update
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .input_embedder import InputEmbedder
from .tick_scheduler import TickScheduler, EvidencePacket
from .lnc_preloader_sim import LNCPreloaderSim
from .self_organizing_allocator import SelfOrganizingAllocator
from .lnc_manager import LNCManager
from .disk_manager import DiskManager
from .system_monitor import SystemMonitor
from .observation_policy import ObservationPolicy
from .efe_proxy import EFEProxyScorer
from .suspension_gate import SuspensionGate
from .energy_kernel import EnergyKernel, EnergyState, ActionProfile


@dataclass
class LearningCycleResult:
    scheduler_output: Dict[str, Any]
    allocator_output: Dict[str, Any]
    committed_actions: Dict[str, List[Any]] = field(default_factory=dict)
    monitor_metrics: Dict[str, float] = field(default_factory=dict)
    kernel_context: Dict[str, Any] = field(default_factory=dict)
    action_dispatch: Dict[str, Any] = field(default_factory=dict)


class ContinualLearningRuntime:
    """显式学习闭环编排器。"""

    def __init__(self,
                 embedder: InputEmbedder,
                 scheduler: TickScheduler,
                 preloader: LNCPreloaderSim,
                 allocator: SelfOrganizingAllocator,
                 lnc_manager: LNCManager,
                 disk_manager: DiskManager,
                 monitor: SystemMonitor,
                 observation_policy: Optional[ObservationPolicy] = None,
                 efe_scorer: Optional[EFEProxyScorer] = None,
                 suspension_gate: Optional[SuspensionGate] = None,
                 energy_kernel: Optional[EnergyKernel] = None,
                 energy_state: Optional[EnergyState] = None):
        self.embedder = embedder
        self.scheduler = scheduler
        self.preloader = preloader
        self.allocator = allocator
        self.lnc_manager = lnc_manager
        self.disk_manager = disk_manager
        self.monitor = monitor
        self.observation_policy = observation_policy or ObservationPolicy()
        self.efe_scorer = efe_scorer or EFEProxyScorer()
        self.suspension_gate = suspension_gate or SuspensionGate()
        self.energy_kernel = energy_kernel or EnergyKernel()
        self.energy_state = energy_state or EnergyState(E=0.8)

    async def run_cycle(self,
                        raw_input: Sequence[float],
                        ts_sense: float,
                        delta_t: float,
                        quality: float,
                        lnc_registry: Dict[str, Dict[str, float]],
                        nt_state: Dict[str, float],
                        lsb_tensor,
                        lnc_table,
                        budget_area: int = 64) -> LearningCycleResult:
        # 1) ObservationPolicy（外循环）
        diagnostics = {
            "residual_energy": float(nt_state.get("residual_energy", 0.0)),
            "uncertainty": float(nt_state.get("uncertainty", 0.2)),
        }
        energy_gates = self.energy_kernel.energy_gating(self.energy_state, precision=float(quality))
        policy_candidates = self.observation_policy.propose_observation_policies(state=nt_state, diagnostics=diagnostics)
        observation = self.observation_policy.select_observation_policy(
            candidates=policy_candidates,
            budget=energy_gates.max_obs_budget,
            state=nt_state,
            diagnostics=diagnostics,
        )

        # 2) Input -> EvidencePacket
        evidence: EvidencePacket = self.embedder.to_evidence_packet(
            raw_input=raw_input,
            delta_t=delta_t,
            ts_sense=ts_sense,
            quality=min(quality, observation.precision_target),
        )

        # 3) 双通道预加载（Routing + Ghost）
        self.preloader.update_evidence(evidence)
        await self.preloader._perform_prediction_cycle()

        # 4) EFE scoring + action pricing
        action_profile = ActionProfile(
            obs_cost=observation.obs_budget_cost,
            inference_cost=0.1 + 0.02 * len(self.preloader.get_last_candidates()),
            plasticity_cost=0.15,
            memory_cost=0.08,
            structure_cost=0.12,
            transport_cost=0.03,
        )
        action_cost = self.energy_kernel.estimate_action_cost(action_profile)
        efe_terms = self.efe_scorer.score(
            prediction_error=float(nt_state.get("prediction_error", diagnostics["residual_energy"])),
            precision=float(evidence.quality),
            complexity=float(len(lnc_registry)) / 10.0,
            instability=float(nt_state.get("pressure", 0.0)),
            resource_cost=action_profile.obs_cost + action_profile.memory_cost,
            action_cost=action_cost,
            energy_available=max(0.0, self.energy_state.E - self.energy_state.E_floor),
        )

        # 5) Scheduler 绑定 + Suspension + Epiphany + Shadow
        preloader_results = self.preloader.get_last_candidates()
        sched_out = self.scheduler.perform_binding(
            lnc_registry=lnc_registry,
            preloader_results=preloader_results,
            nt_state=nt_state,
            evidence_packet=evidence,
            energy_gates=energy_gates.__dict__,
            efe_score=efe_terms.total,
        )

        suspension = self.suspension_gate.decide(
            efe_score=efe_terms.total,
            mode=sched_out.get("mode", "idle_drift"),
            precision=float(evidence.quality),
            resonance={
                "ghost_peak": 1.0 if sched_out.get("epiphany_signal", {}).get("triggered") else 0.0,
                "curvature_peak": float(nt_state.get("curvature", 0.0)),
            },
            resource={"energy": self.energy_state.E, "pressure": float(nt_state.get("pressure", 0.0))},
            energy_gates=energy_gates.__dict__,
        )
        sched_out["suspension_action"] = suspension.action.value
        sched_out["suspension_reason"] = suspension.reason

        # 6) replay window 消费（低频巩固）
        replay_batch = []
        replay_cfg = sched_out.get("replay_window", {})
        if replay_cfg.get("enabled", False):
            replay_batch = self.disk_manager.consume_replay_batch(replay_cfg.get("max_items", 2))

        # 7) allocator 执行局部尺度调节 + 结构动作提议
        alloc_out = self.allocator.run_allocation_cycle(
            lsb_tensor=lsb_tensor,
            lnc_table=lnc_table,
            budget_area=budget_area,
            epiphany_signal=sched_out.get("epiphany_signal"),
            shadow_plan=sched_out.get("shadow_plan"),
            replay_batch=replay_batch,
            energy_gates=energy_gates.__dict__,
        )

        # 8) 结构动作提交（split / merge）
        committed = {"split": [], "merge": [], "rollback": [], "snapshot": []}
        structure_actions = alloc_out.get("structure_actions", {})

        if energy_gates.allow_structure_split:
            for c in structure_actions.get("split_candidates", []):
                lid = int(c.get("lnc_id", -1))
                if lid >= 0:
                    new_id = self.lnc_manager.split_lnc_with_shadow(lid)
                    if new_id is not None:
                        committed["split"].append((lid, new_id))
                        self.monitor.log_event("split_committed", {"parent": lid, "child": new_id})

        for hint in sched_out.get("structure_plan", {}).get("merge_hints", []):
            parent = hint.get("parent")
            if isinstance(parent, str) and parent.isdigit():
                pid = int(parent)
                node = self.lnc_manager.blanket_hierarchy.get(pid, {})
                children = node.get("children", [])
                if children:
                    loser = int(children[0])
                    if self.lnc_manager.merge_lncs(pid, loser):
                        committed["merge"].append((pid, loser))
                        self.monitor.log_event("merge_committed", {"winner": pid, "loser": loser})

        if energy_gates.allow_snapshot:
            for lid_str in sched_out.get("selected_lncs", [])[:2]:
                if isinstance(lid_str, str) and lid_str.isdigit():
                    lid = int(lid_str)
                    payload = self.lnc_manager.save_structured_snapshot(lid, disk_manager=self.disk_manager)
                    if payload is not None:
                        committed["snapshot"].append(lid)

        if sched_out.get("residual_energy_proxy", 0.0) > 0.9:
            for lid_str in sched_out.get("selected_lncs", [])[:1]:
                if isinstance(lid_str, str) and lid_str.isdigit():
                    lid = int(lid_str)
                    if self.lnc_manager.rollback_structured_snapshot(lid, disk_manager=self.disk_manager):
                        committed["rollback"].append(lid)
                        self.monitor.log_event("rollback_applied", {"lnc_id": lid})

        for lid_str in sched_out.get("selected_lncs", [])[:2]:
            self.disk_manager.update_transport_metric(f"{lid_str}->global", alignment=0.6, curvature=0.2)
        diag = self.disk_manager.transport_diagnostics()

        # 9) 能量动力学回写
        self.energy_state = self.energy_kernel.apply_energy_dynamics(
            self.energy_state,
            action_cost=action_cost,
            diagnostics={
                "residual_energy": sched_out.get("residual_energy_proxy", 0.0),
                "precision": evidence.quality,
                "ts": ts_sense,
            },
        )

        # 10) monitor
        trace_only_rate = 1.0 if suspension.action.value in ("TRACE_ONLY", "SLEEP") else 0.0
        update_rate = 1.0 if suspension.action.value == "UPDATE" else 0.0
        metrics = self.monitor.record_continual_metrics(
            lsb_tensor=lsb_tensor,
            scheduler_output=sched_out,
            ghost_count=len(getattr(self.disk_manager, "_ghosts", {})),
            epiphany_events=1 if sched_out.get("epiphany_signal", {}).get("triggered") else 0,
            compute_budget=float(alloc_out.get("budget_spent", 0.0)),
            key_drift=0.0,
            transport_curvature=float(diag.get("curvature_mean", 0.0)),
            energy_E=self.energy_state.E,
            stress=self.energy_state.stress,
            arousal=self.energy_state.arousal,
            trace_only_rate=trace_only_rate,
            update_rate=update_rate,
            obs_budget_spent=observation.obs_budget_cost,
        )

        return LearningCycleResult(
            scheduler_output=sched_out,
            allocator_output=alloc_out,
            committed_actions=committed,
            monitor_metrics=metrics,
            kernel_context=kernel_context,
            action_dispatch=action_dispatch,
        )
