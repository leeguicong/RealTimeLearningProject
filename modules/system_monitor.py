#
# 文件: modules/system_monitor.py
# 职责: 系统的“核磁共振仪” (fMRI) + 持续学习指标监控。

import os
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import neuron_schema as schema


class SystemMonitor:
    """系统监视器。"""

    def __init__(self, output_dir="monitor_logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"SystemMonitor: Initialized. Saving viz to '{output_dir}/'")

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle("AI Brain Activity Monitor")

        self.metric_history: Dict[str, List[float]] = {
            "routing_entropy": [],
            "residual_energy": [],
            "ghost_count": [],
            "epiphany_events": [],
            "compute_budget": [],
            "shadow_branches": [],
            "key_drift": [],
            "curvature": [],
            "energy_E": [],
            "stress": [],
            "arousal": [],
            "trace_only_rate": [],
            "update_rate": [],
            "obs_budget_spent": [],
        }
        self.event_log: List[Dict[str, Any]] = []

    def capture_snapshot(self, lsb_tensor: torch.Tensor, tick: int):
        step = 8
        activity_map = lsb_tensor[::step, ::step, schema.AMPLITUDE_INDEX].cpu().float().numpy()
        chem_map = lsb_tensor[::step, ::step, schema.NT_AMPLITUDE_INDEX].cpu().float().numpy()
        surprise_map = lsb_tensor[::step, ::step, schema.PREDICTION_ERROR_INDEX].cpu().float().numpy()

        self._plot_heatmap(self.axes[0], activity_map, "Neural Activity (Amplitude)", cmap="viridis")
        self._plot_heatmap(self.axes[1], chem_map, "Neurotransmitter (Attention)", cmap="plasma")
        self._plot_heatmap(self.axes[2], surprise_map, "Prediction Error (Surprise)", cmap="Reds")

        save_path = os.path.join(self.output_dir, f"tick_{tick:06d}.png")
        plt.savefig(save_path)
        print(f"SystemMonitor: Snapshot saved to {save_path}")

    def record_continual_metrics(self,
                                 lsb_tensor: torch.Tensor,
                                 scheduler_output: Dict[str, Any],
                                 ghost_count: int = 0,
                                 epiphany_events: int = 0,
                                 compute_budget: float = 0.0,
                                 key_drift: float = 0.0,
                                 transport_curvature: float = 0.0,
                                 energy_E: float = 0.0,
                                 stress: float = 0.0,
                                 arousal: float = 0.0,
                                 trace_only_rate: float = 0.0,
                                 update_rate: float = 0.0,
                                 obs_budget_spent: float = 0.0) -> Dict[str, float]:
        """记录持续学习关键指标（Phase1 最小闭环）。"""
        routing_entropy = float(scheduler_output.get("routing_entropy_proxy", 0.0))
        residual_energy = float(torch.mean(torch.abs(lsb_tensor[..., schema.RESIDUAL_ENERGY_INDEX])).item())

        self.metric_history["routing_entropy"].append(routing_entropy)
        self.metric_history["residual_energy"].append(residual_energy)
        self.metric_history["ghost_count"].append(float(ghost_count))
        self.metric_history["epiphany_events"].append(float(epiphany_events))
        self.metric_history["compute_budget"].append(float(compute_budget))
        shadow_count = len(scheduler_output.get("shadow_plan", {}).get("active_branches", []))
        self.metric_history["shadow_branches"].append(float(shadow_count))
        self.metric_history["key_drift"].append(float(key_drift))
        self.metric_history["curvature"].append(float(transport_curvature))
        self.metric_history["energy_E"].append(float(energy_E))
        self.metric_history["stress"].append(float(stress))
        self.metric_history["arousal"].append(float(arousal))
        self.metric_history["trace_only_rate"].append(float(trace_only_rate))
        self.metric_history["update_rate"].append(float(update_rate))
        self.metric_history["obs_budget_spent"].append(float(obs_budget_spent))

        return {
            "routing_entropy": routing_entropy,
            "residual_energy": residual_energy,
            "ghost_count": float(ghost_count),
            "epiphany_events": float(epiphany_events),
            "compute_budget": float(compute_budget),
            "shadow_branches": float(shadow_count),
            "key_drift": float(key_drift),
            "curvature": float(transport_curvature),
            "energy_E": float(energy_E),
            "stress": float(stress),
            "arousal": float(arousal),
            "trace_only_rate": float(trace_only_rate),
            "update_rate": float(update_rate),
            "obs_budget_spent": float(obs_budget_spent),
        }

    def save_metric_plot(self, tick: int) -> str:
        """保存关键指标曲线。"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.metric_history["routing_entropy"], label="routing_entropy")
        axes[0].plot(self.metric_history["residual_energy"], label="residual_energy")
        axes[0].legend()
        axes[0].set_title("Routing / Residual")

        axes[1].plot(self.metric_history["ghost_count"], label="ghost_count")
        axes[1].plot(self.metric_history["epiphany_events"], label="epiphany_events")
        axes[1].plot(self.metric_history["compute_budget"], label="compute_budget")
        axes[1].plot(self.metric_history["shadow_branches"], label="shadow_branches")
        axes[1].plot(self.metric_history["key_drift"], label="key_drift")
        axes[1].plot(self.metric_history["curvature"], label="curvature")
        axes[1].plot(self.metric_history["energy_E"], label="energy_E")
        axes[1].plot(self.metric_history["stress"], label="stress")
        axes[1].plot(self.metric_history["arousal"], label="arousal")
        axes[1].plot(self.metric_history["trace_only_rate"], label="trace_only_rate")
        axes[1].plot(self.metric_history["update_rate"], label="update_rate")
        axes[1].plot(self.metric_history["obs_budget_spent"], label="obs_budget_spent")
        axes[1].legend()
        axes[1].set_title("Ghost / Events / Budget")

        save_path = os.path.join(self.output_dir, f"metrics_{tick:06d}.png")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.event_log.append({"event": event_type, "payload": payload})

    def _plot_heatmap(self, ax, data, title, cmap):
        ax.clear()
        ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=0, vmax=1.0)
        ax.set_title(title)
        ax.axis('off')
