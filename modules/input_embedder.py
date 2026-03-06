"""Input embedder and evidence acquisition contracts."""

from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .tick_scheduler import EvidencePacket


@dataclass
class EmbeddedQuery:
    q_key: List[float]
    q_value: List[float]
    q_residual_sig: List[float]


class InputEmbedder:
    """将原始输入转为 EvidencePacket（e_key/e_val/e_res_hint）。"""

    def __init__(self, key_dim: int = 3, value_dim: int = 6):
        self.key_dim = key_dim
        self.value_dim = value_dim

    def embed(self, raw_input: Sequence[float], ts_sense: float = 0.0) -> EmbeddedQuery:
        vals = [float(x) for x in raw_input] or [0.0]

        q_value = (vals + [0.0] * self.value_dim)[: self.value_dim]
        mean_v = sum(vals) / len(vals)
        rms_v = math.sqrt(sum(v * v for v in vals) / len(vals))
        span_v = max(vals) - min(vals)
        q_key = ([mean_v, rms_v, span_v] + [0.0] * self.key_dim)[: self.key_dim]

        centered = [v - mean_v for v in vals]
        if len(centered) < self.key_dim:
            centered += [0.0] * (self.key_dim - len(centered))
        q_residual_sig = centered[: self.key_dim]

        return EmbeddedQuery(q_key=q_key, q_value=q_value, q_residual_sig=q_residual_sig)

    def to_evidence_packet(self,
                           raw_input: Sequence[float],
                           delta_t: float,
                           ts_sense: float,
                           quality: float = 1.0) -> EvidencePacket:
        e = self.embed(raw_input, ts_sense=ts_sense)
        return EvidencePacket(
            e_key=e.q_key,
            e_val=e.q_value,
            e_rel={"delta_t": float(delta_t)},
            e_res_hint=e.q_residual_sig,
            ts_sense=float(ts_sense),
            quality=float(quality),
        )


class IInputEmbedder(abc.ABC):
    @abc.abstractmethod
    def initialize_sensor(self, sensor_config: Dict[str, Any]) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def emit_evidence(self, sensor_state: Optional[Dict[str, Any]] = None) -> List[EvidencePacket]:
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError


class SimulatedGaborEmbedder(IInputEmbedder):
    """兼容旧导出接口的简化模拟输入器。"""

    def __init__(self, source_id: str = "sim"):
        self.source_id = source_id
        self.initialized = False

    def initialize_sensor(self, sensor_config: Dict[str, Any]) -> bool:
        self.initialized = True
        return True

    async def emit_evidence(self, sensor_state: Optional[Dict[str, Any]] = None) -> List[EvidencePacket]:
        sensor_state = sensor_state or {}
        values = sensor_state.get("values", [0.0, 0.0, 0.0])
        quality = float(sensor_state.get("quality", 1.0))
        delta_t = float(sensor_state.get("delta_t", 0.1))
        ts = float(sensor_state.get("timestamp", time.time()))
        packet = InputEmbedder().to_evidence_packet(values, delta_t=delta_t, ts_sense=ts, quality=quality)
        return [packet]

    def execute_and_transfer(self, target_tensor_on_chip_a: torch.Tensor) -> None:
        if target_tensor_on_chip_a.numel() > 0:
            target_tensor_on_chip_a.view(-1)[0] = 1.0

    def shutdown(self) -> None:
        self.initialized = False
