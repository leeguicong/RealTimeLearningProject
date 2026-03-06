"""DiskManager (Phase2 minimal implementation).

职责：
- 提供余弦相似检索（query_resonance）
- 提供 Ghost Store 持久索引（内存模拟）
- 提供 ghost 共振查询与衰减
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import copy


def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    aa = a[:n]
    bb = b[:n]
    na = _l2_norm(aa)
    nb = _l2_norm(bb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(aa, bb)) / (na * nb)


@dataclass
class GhostEntry:
    ghost_id: str
    sig: List[float]
    linked_lnc_ids: List[str]
    energy: float
    ttl: int = 100
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class SnapshotRecord:
    version: int
    lnc_id: str
    payload: Dict[str, object]


class DiskManager:
    def __init__(self):
        self._lnc_fingerprints: Dict[str, List[float]] = {}
        self._ghosts: Dict[str, GhostEntry] = {}
        self._ghost_by_sig_bucket: Dict[int, List[str]] = {}
        self._snapshot_versions: Dict[str, List[SnapshotRecord]] = {}
        self._replay_queue: List[Dict[str, object]] = []
        self._transport_metrics: Dict[str, Dict[str, float]] = {}

    async def query_resonance(self, query_vector: List[float], top_k: int = 3) -> List[str]:
        scored: List[Tuple[str, float]] = []
        for lnc_id, fp in self._lnc_fingerprints.items():
            scored.append((lnc_id, _cosine(query_vector, fp)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [lid for lid, score in scored[:top_k] if score > 0.0]

    def upsert_lnc_fingerprint(self, lnc_id: str, fingerprint: List[float]) -> None:
        self._lnc_fingerprints[lnc_id] = list(fingerprint)

    def put_ghost(self, entry: GhostEntry) -> None:
        self._ghosts[entry.ghost_id] = entry
        bucket = self._bucket_sig(entry.sig)
        self._ghost_by_sig_bucket.setdefault(bucket, [])
        if entry.ghost_id not in self._ghost_by_sig_bucket[bucket]:
            self._ghost_by_sig_bucket[bucket].append(entry.ghost_id)

    def query_ghost_resonance(self, sig: List[float], top_k: int = 3) -> List[GhostEntry]:
        bucket = self._bucket_sig(sig)
        candidates = self._ghost_by_sig_bucket.get(bucket, list(self._ghosts.keys()))
        scored: List[Tuple[str, float]] = []
        for gid in candidates:
            ghost = self._ghosts.get(gid)
            if ghost is None:
                continue
            score = _cosine(sig, ghost.sig) * max(0.0, ghost.energy)
            scored.append((gid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        out: List[GhostEntry] = []
        for gid, score in scored[:top_k]:
            if score <= 0.0:
                continue
            ghost = self._ghosts.get(gid)
            if ghost is not None:
                out.append(ghost)
        return out

    def decay_ghosts(self, decay: float = 0.98) -> None:
        to_delete: List[str] = []
        for gid, ghost in self._ghosts.items():
            ghost.energy *= decay
            ghost.ttl -= 1
            if ghost.ttl <= 0 or ghost.energy < 1e-3:
                to_delete.append(gid)
        for gid in to_delete:
            del self._ghosts[gid]
            for bucket_ids in self._ghost_by_sig_bucket.values():
                if gid in bucket_ids:
                    bucket_ids.remove(gid)

    def _bucket_sig(self, sig: List[float]) -> int:
        if not sig:
            return 0
        return int((sum(sig) / len(sig)) * 10)

    def save_structured_snapshot(self, lnc_id: str, payload: Dict[str, object]) -> int:
        versions = self._snapshot_versions.setdefault(lnc_id, [])
        next_version = versions[-1].version + 1 if versions else 1
        versions.append(SnapshotRecord(next_version, lnc_id, copy.deepcopy(payload)))
        return next_version

    def rollback_snapshot(self, lnc_id: str, target_version: int | None = None) -> Dict[str, object] | None:
        versions = self._snapshot_versions.get(lnc_id, [])
        if not versions:
            return None
        if target_version is None:
            target = versions[-2] if len(versions) > 1 else versions[-1]
        else:
            matched = [v for v in versions if v.version == target_version]
            if not matched:
                return None
            target = matched[-1]
        versions.append(SnapshotRecord(versions[-1].version + 1, lnc_id, copy.deepcopy(target.payload)))
        return copy.deepcopy(target.payload)

    def enqueue_replay_trace(self, trace_item: Dict[str, object]) -> None:
        self._replay_queue.append(trace_item)

    def consume_replay_batch(self, max_items: int = 2) -> List[Dict[str, object]]:
        if max_items <= 0:
            return []
        batch = self._replay_queue[:max_items]
        self._replay_queue = self._replay_queue[max_items:]
        return batch

    def update_transport_metric(self, edge_key: str, alignment: float, curvature: float) -> None:
        """记录跨LNC transport 对齐质量（connection/1-form/2-form 占位）。"""
        self._transport_metrics[edge_key] = {
            "alignment": float(alignment),
            "curvature": float(curvature),
        }

    def get_transport_metric(self, edge_key: str) -> Dict[str, float]:
        return dict(self._transport_metrics.get(edge_key, {"alignment": 0.0, "curvature": 0.0}))

    def transport_diagnostics(self) -> Dict[str, float]:
        if not self._transport_metrics:
            return {"alignment_mean": 0.0, "curvature_mean": 0.0, "edges": 0.0}
        vals = list(self._transport_metrics.values())
        align_mean = sum(v["alignment"] for v in vals) / len(vals)
        curv_mean = sum(v["curvature"] for v in vals) / len(vals)
        return {"alignment_mean": align_mean, "curvature_mean": curv_mean, "edges": float(len(vals))}
