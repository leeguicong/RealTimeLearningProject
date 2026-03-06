#
# 文件: modules/lnc_preloader_sim.py
# 职责: 模拟 "LNC-Preloader" (预测性加载器)。
#
# 改造点（Phase 1）:
# - 引入双通道检索：key 通道 + residual/ghost 通道（概念对齐）
# - 接收 EvidencePacket（若未提供则回退到化学状态构造）

import asyncio
import numpy as np
from typing import List, Optional

from .lnc_manager import LNCManager
from .disk_manager import DiskManager
from .neurotransmitter_manager import NeurotransmitterManager
from .tick_scheduler import EvidencePacket


class LNCPreloaderSim:
    """LNC 预测加载器 (模拟器)。"""

    def __init__(self,
                 lnc_manager: LNCManager,
                 disk_manager: DiskManager,
                 nt_manager: NeurotransmitterManager,
                 check_interval: float = 1.0):
        print("LNCPreloaderSim: Initializing (Subconscious Intuition)...")
        self.lnc_manager = lnc_manager
        self.disk_manager = disk_manager
        self.nt_manager = nt_manager
        self.check_interval = check_interval
        self._is_running = False
        self._latest_evidence: Optional[EvidencePacket] = None
        self._last_candidate_ids: List[str] = []

    def update_evidence(self, evidence_packet: EvidencePacket) -> None:
        """由调度器/输入模块注入最新 EvidencePacket。"""
        self._latest_evidence = evidence_packet

    async def start(self):
        self._is_running = True
        print("LNCPreloaderSim: Started background prediction loop.")

        while self._is_running:
            try:
                await self._perform_prediction_cycle()
            except Exception as e:
                print(f"LNCPreloaderSim Error: {e}")
            await asyncio.sleep(self.check_interval)

    def stop(self):
        self._is_running = False

    def get_last_candidates(self) -> List[str]:
        return list(self._last_candidate_ids)

    async def _fallback_packet_from_chemistry(self) -> EvidencePacket:
        chem_state = await self.nt_manager.get_current_state()
        valance = np.cos(chem_state.get("GLOBAL_VALUE", 0.0) * np.pi)
        arousal = chem_state.get("GLOBAL_METABOLIC", 1.0)
        surprise = chem_state.get("GLOBAL_SURPRISE", 0.0)
        return EvidencePacket(
            e_key=[arousal, float(valance), surprise],
            e_val=[arousal, surprise],
            e_rel={"delta_t": chem_state.get("DELTA_T", 0.0)},
            e_res_hint=[surprise],
            ts_sense=chem_state.get("TS", 0.0),
            quality=chem_state.get("QUALITY", 1.0),
        )

    async def _perform_prediction_cycle(self):
        evidence = self._latest_evidence or await self._fallback_packet_from_chemistry()

        # 通道1: key 召回候选 LNC
        key_candidates = await self.disk_manager.query_resonance(evidence.e_key, top_k=3)

        # 通道2: residual hint 召回 ghost，并转为关联 LNC 候选
        residual_query = evidence.e_res_hint or evidence.e_key
        ghost_hits = self.disk_manager.query_ghost_resonance(residual_query, top_k=3)
        residual_candidates: List[str] = []
        for ghost in ghost_hits:
            for linked in ghost.linked_lnc_ids:
                if linked not in residual_candidates:
                    residual_candidates.append(linked)

        # 探索预算：低质量/高残差时，放宽 key 通道召回数量
        res_energy = float(sum(abs(x) for x in residual_query))
        if evidence.quality < 0.5 or res_energy > 0.8:
            explore_candidates = await self.disk_manager.query_resonance(evidence.e_key, top_k=5)
            for lnc_id in explore_candidates:
                if lnc_id not in key_candidates:
                    key_candidates.append(lnc_id)

        candidate_ids: List[str] = []
        for lnc_id in key_candidates + residual_candidates:
            if lnc_id not in candidate_ids:
                candidate_ids.append(lnc_id)

        self._last_candidate_ids = list(candidate_ids)

        if not candidate_ids:
            return

        for lnc_id in candidate_ids:
            lnc_id_int = hash(lnc_id) % 2**16
            if lnc_id_int not in self.lnc_manager.l1_active_lncs:
                print(f"LNCPreloaderSim: [PREDICTION] 双通道召回 {lnc_id}，执行预加载...")
                success = self.lnc_manager._load_lnc_to_l1(lnc_id)
                if success:
                    print(f"LNCPreloaderSim: -> {lnc_id} 预加载成功。")
