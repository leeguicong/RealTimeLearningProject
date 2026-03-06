# main.py
# 系统编排层：装配模块、维护生命周期、驱动三类循环
# 认知主循环 = ContinualLearningRuntime
# main.py 不直接实现认知逻辑，只负责 orchestration / bridging / IO

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from modules import neuron_schema as schema
from modules.large_static_buffer import LargeStaticBuffer
from modules.physics_kernel_v1 import PhysicsKernelV1
from modules.neurotransmitter_manager import NeurotransmitterManager
from modules.lnc_manager import LNCManager
from modules.lnc_preloader_sim import LNCPreloaderSim
from modules.disk_manager import DiskManager
from modules.system_monitor import SystemMonitor
from modules.input_embedder import InputEmbedder, SimulatedGaborEmbedder
from modules.output_controller import SimulatedMotorController, IOutputController
from modules.tick_scheduler import TickScheduler, EvidencePacket
from modules.self_organizing_allocator import SelfOrganizingAllocator
from modules.continual_learning_runtime import ContinualLearningRuntime
from modules.energy_kernel import EnergyKernel, EnergyState


# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------

@dataclass
class AppConfig:
    target_tick_interval_s: float = 0.005
    maintenance_interval_ticks: int = 100
    monitor_interval_ticks: int = 100
    consolidation_interval_ticks: int = 500
    action_region_fallback_hw: Tuple[int, int] = (7, 2)
    genesis_list: List[str] = field(default_factory=lambda: ["LNC-V1-Skin", "LNC-Planner"])
    budget_area: int = 64
    monitor_output_dir: str = "monitor_logs"


# ---------------------------------------------------------------------
# 轻量消息结构
# ---------------------------------------------------------------------

@dataclass
class SensorFrame:
    raw_input: Sequence[float]
    ts_sense: float
    quality: float = 1.0


@dataclass
class CognitiveStepResult:
    evidence: Optional[EvidencePacket]
    runtime_result: Any
    delta_t: float
    tick_id: int


# ---------------------------------------------------------------------
# Bridge: 认知层 <-> 物理层
# ---------------------------------------------------------------------

class RuntimeKernelBridge:
    """
    将认知层输出映射到物理层 kernel 执行参数与 LSB action slice。
    这是现在系统最需要补齐的“中间层”。
    """

    def __init__(self, lsb: LargeStaticBuffer, lnc_manager: LNCManager, config: AppConfig):
        self.lsb = lsb
        self.lnc_manager = lnc_manager
        self.config = config

    def build_kernel_context(
        self,
        evidence: Optional[EvidencePacket],
        runtime_result: Optional[Any],
        default_dt: float,
    ) -> Dict[str, Any]:
        """
        目前先给出最小桥接：
        - dt 受 scheduler mode / suspension action 影响
        - raw sensory value 写入单独 sensor tensor
        - 后续可继续扩展 trace_only_mask / active_region_mask / dt_scale_table
        """
        dt = default_dt
        sensor_tensor = None

        if evidence is not None:
            vals = list(evidence.e_val)
            sensor_tensor = torch.tensor(vals, dtype=torch.float32, device=self.lsb.device)

        if runtime_result is not None:
            sched = getattr(runtime_result, "scheduler_output", {}) or {}
            mode = sched.get("mode", "idle_drift")
            suspension = sched.get("suspension_action", "update")

            if mode == "sleep":
                dt = default_dt * 0.25
            elif mode == "idle_drift":
                dt = default_dt * 0.5
            elif mode == "active_burst":
                dt = default_dt

            if suspension == "trace_only":
                dt *= 0.5

        return {
            "dt": dt,
            "sensor_tensor": sensor_tensor,
        }

    def extract_action_tensor(self, output_spec: Tuple[Tuple[int, ...], torch.dtype]) -> torch.Tensor:
        """
        目前实现两级策略：
        1. 若以后 LNCManager 提供角色绑定，就优先走 role='motor'
        2. 否则走 fallback 区域
        """
        shape, dtype = output_spec

        # ---- 预留：未来正式替换为 role-based action routing ----
        # e.g. region = self.lnc_manager.get_role_bound_region("motor")
        region = None

        if region is not None:
            px, py, pw, ph = region
            action_slice = self.lsb.get_slice(px, py, pw, ph)
            return action_slice[..., : shape[-1]].reshape(shape).to(dtype=dtype)

        # ---- fallback: 沿用旧 main 的最小兼容策略，但集中封装在 bridge 内 ----
        h, w = self.config.action_region_fallback_hw
        raw = self.lsb._buffer[0:h, 0:1, 0:w]
        return raw.reshape(shape).to(dtype=dtype)


# ---------------------------------------------------------------------
# 主应用
# ---------------------------------------------------------------------

class CognitiveSystemApp:
    """
    main.py 的新角色：
    - 只做编排与桥接
    - 不直接实现认知逻辑
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- 基础模块 ----------
        self.lsb = LargeStaticBuffer()
        self.nt_manager = NeurotransmitterManager()
        self.lnc_manager = LNCManager(self.lsb, bus=None)
        self.disk_manager = getattr(self.lnc_manager, "disk_manager", None) or DiskManager()
        self.preloader = LNCPreloaderSim(
            lnc_manager=self.lnc_manager,
            disk_manager=self.disk_manager,
            nt_manager=self.nt_manager,
        )
        self.kernel = PhysicsKernelV1(
            k_coupling_const=0.1,
            energy_replenish_rate=0.01,
            base_learning_rate=0.01,
        )
        self.monitor = SystemMonitor(output_dir=self.config.monitor_output_dir)

        # ---------- 认知模块 ----------
        self.embedder = InputEmbedder()
        self.scheduler = TickScheduler()
        self.allocator = SelfOrganizingAllocator()
        self.energy_kernel = EnergyKernel()
        self.energy_state = EnergyState(E=0.8)
        self.runtime = ContinualLearningRuntime(
            embedder=self.embedder,
            scheduler=self.scheduler,
            preloader=self.preloader,
            allocator=self.allocator,
            lnc_manager=self.lnc_manager,
            disk_manager=self.disk_manager,
            monitor=self.monitor,
            energy_kernel=self.energy_kernel,
            energy_state=self.energy_state,
        )

        # ---------- IO 适配器 ----------
        self.input_driver = SimulatedGaborEmbedder(source_id="sim")
        self.output_driver: IOutputController = SimulatedMotorController(
            sim_spec=((7, 2), torch.float16),
            sim_physical_constants=torch.tensor([0.5] * 7, dtype=torch.float16),
        )

        # ---------- bridge ----------
        self.bridge = RuntimeKernelBridge(self.lsb, self.lnc_manager, self.config)

        # ---------- 运行态 ----------
        self.tick_counter = 0
        self.is_running = False
        self._tasks: List[asyncio.Task] = []

        # 通道
        self.sensor_queue: asyncio.Queue[SensorFrame] = asyncio.Queue(maxsize=32)
        self.cognitive_queue: asyncio.Queue[CognitiveStepResult] = asyncio.Queue(maxsize=32)

        self._last_tick_ts = time.perf_counter()

    # -----------------------------------------------------------------
    # Bootstrap / Shutdown
    # -----------------------------------------------------------------

    async def bootstrap(self) -> None:
        print("=" * 60)
        print("CognitiveSystemApp: Booting...")
        print("=" * 60)
        print(f"Device: {self.device}")

        self.lsb.allocate()
        self._inject_initial_noise()

        # 初始化输入/输出适配器
        self.input_driver.initialize_sensor({})
        self.output_driver.initialize_controller({})

        # 身体常量可以在这里注入到系统中，作为启动时的“身体模型”观测
        body_constants = self.output_driver.get_physical_constants()
        print(f"Body constants loaded: shape={tuple(body_constants.shape)}")

        # 创世
        self.lnc_manager.start_genesis(self.config.genesis_list)

        self.is_running = True

    async def shutdown(self) -> None:
        self.is_running = False

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        try:
            self.output_driver.shutdown()
        except Exception:
            pass

        try:
            self.input_driver.shutdown()
        except Exception:
            pass

        print("CognitiveSystemApp: Shutdown complete.")

    # -----------------------------------------------------------------
    # Loop launcher
    # -----------------------------------------------------------------

    async def run(self) -> None:
        if not self.is_running:
            raise RuntimeError("Call bootstrap() before run().")

        self._tasks = [
            asyncio.create_task(self.sensor_loop(), name="sensor_loop"),
            asyncio.create_task(self.cognitive_loop(), name="cognitive_loop"),
            asyncio.create_task(self.action_loop(), name="action_loop"),
            asyncio.create_task(self.maintenance_loop(), name="maintenance_loop"),
        ]

        await asyncio.gather(*self._tasks)

    # -----------------------------------------------------------------
    # Sensor loop
    # -----------------------------------------------------------------

    async def sensor_loop(self) -> None:
        """
        感官循环：
        - 从各模态适配器取样
        - 统一成 SensorFrame
        - 送入认知队列
        """
        while self.is_running:
            ts = time.perf_counter()

            # 这里优先走新版接口：emit_evidence
            # 如果未来有真正的多模态输入器，可直接在这里做 batch / timestamp align
            packets = await self.input_driver.emit_evidence(sensor_state=None)

            if not packets:
                await asyncio.sleep(self.config.target_tick_interval_s)
                continue

            for pkt in packets:
                frame = SensorFrame(
                    raw_input=list(pkt.e_val),
                    ts_sense=float(pkt.ts_sense),
                    quality=float(pkt.quality),
                )
                await self.sensor_queue.put(frame)

            await asyncio.sleep(self.config.target_tick_interval_s)

    # -----------------------------------------------------------------
    # Cognitive loop
    # -----------------------------------------------------------------

    async def cognitive_loop(self) -> None:
        """
        唯一认知主循环：
        - 消费 sensor frames
        - 调 runtime.run_cycle()
        - 把结果送给 action loop
        """
        while self.is_running:
            frame = await self.sensor_queue.get()

            now = time.perf_counter()
            delta_t = now - self._last_tick_ts
            self._last_tick_ts = now
            self.tick_counter += 1

            nt_state = self._collect_nt_state()
            lnc_registry = self._build_lnc_registry()
            lsb_tensor = self.lsb._buffer
            lnc_table = self._build_lnc_table_stub()

            runtime_result = await self.runtime.run_cycle(
                raw_input=frame.raw_input,
                ts_sense=frame.ts_sense,
                delta_t=delta_t,
                quality=frame.quality,
                lnc_registry=lnc_registry,
                nt_state=nt_state,
                lsb_tensor=lsb_tensor,
                lnc_table=lnc_table,
                budget_area=self.config.budget_area,
            )

            evidence = self.embedder.to_evidence_packet(
                raw_input=frame.raw_input,
                delta_t=delta_t,
                ts_sense=frame.ts_sense,
                quality=frame.quality,
            )

            # 认知结果送给动作/物理层
            await self.cognitive_queue.put(
                CognitiveStepResult(
                    evidence=evidence,
                    runtime_result=runtime_result,
                    delta_t=delta_t,
                    tick_id=self.tick_counter,
                )
            )

    # -----------------------------------------------------------------
    # Action / Physics loop
    # -----------------------------------------------------------------

    async def action_loop(self) -> None:
        """
        物理-动作执行循环：
        - 根据认知结果构造 kernel context
        - 推进一步物理世界
        - 提取 action tensor
        - 发给外部动作协处理器
        """
        output_spec = self.output_driver.get_specification()

        while self.is_running:
            step = await self.cognitive_queue.get()

            kernel_ctx = self.bridge.build_kernel_context(
                evidence=step.evidence,
                runtime_result=step.runtime_result,
                default_dt=self.config.target_tick_interval_s,
            )

            sensor_tensor = kernel_ctx["sensor_tensor"]
            dt = float(kernel_ctx["dt"])

            # 最小兼容：若 kernel 仍要求 sensor_input_tensor，则把 EvidencePacket.e_val 送进去
            if sensor_tensor is None:
                sensor_tensor = torch.zeros((self.embedder.value_dim,), dtype=torch.float32, device=self.device)

            self.kernel.execute(
                self.lsb._buffer,
                sensor_tensor,
                dt=dt,
            )

            action_tensor = self.bridge.extract_action_tensor(output_spec)
            await self.output_driver.execute_oscillator_command(action_tensor)

    # -----------------------------------------------------------------
    # Maintenance loop
    # -----------------------------------------------------------------

    async def maintenance_loop(self) -> None:
        """
        慢循环：
        - monitor
        - consolidation
        - snapshot / replay
        - 未来可放 health check / drift correction
        """
        while self.is_running:
            tick = self.tick_counter

            if tick > 0 and tick % self.config.monitor_interval_ticks == 0:
                self._run_monitor()

            if tick > 0 and tick % self.config.consolidation_interval_ticks == 0:
                self.lnc_manager.process_pending_consolidation(max_items=2)

            await asyncio.sleep(self.config.target_tick_interval_s * 10)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _inject_initial_noise(self) -> None:
        buffer = self.lsb._buffer
        if buffer is None:
            return

        buffer[..., schema.BELIEF_SHAPE_K_INDEX].normal_(mean=1.0, std=0.1).abs_()
        buffer[..., schema.BELIEF_RATE_THETA_INDEX].normal_(mean=1.0, std=0.1).abs_()
        buffer[..., schema.AMPLITUDE_INDEX].normal_(mean=0.01, std=0.005).abs_()
        buffer[..., schema.PHASE_INDEX].uniform_(0, 2 * torch.pi)

    def _collect_nt_state(self) -> Dict[str, float]:
        """
        先给最小 stub。后续应由 NT manager / monitor / kernel diagnostics 回填。
        """
        return {
            "prediction_error": 0.1,
            "residual_energy": 0.1,
            "uncertainty": 0.2,
            "pressure": 0.1,
            "curvature": 0.0,
            "energy_budget": max(0.0, self.runtime.energy_state.E - self.runtime.energy_state.E_floor),
        }

    def _build_lnc_registry(self) -> Dict[str, Dict[str, float]]:
        registry: Dict[str, Dict[str, float]] = {}
        for lid in self.lnc_manager.l1_active_lncs.keys():
            registry[str(lid)] = {
                "active": 1.0,
                "salience": 1.0,
            }
        return registry

    def _build_lnc_table_stub(self):
        """
        allocator/run_cycle 需要 lnc_table。
        在现有工程里，这个表的正式版以后应来自 lnc_schema + lnc_manager registry。
        现在先给最小占位，保持主循环闭合。
        """
        n = max(1, len(self.lnc_manager.l1_active_lncs))
        return torch.zeros((n, 16), dtype=torch.float32, device=self.device)

    def _run_monitor(self) -> None:
        try:
            # 这里保留成最小入口，按你的 monitor API 再细化
            print(f"[Monitor] tick={self.tick_counter}, active_lncs={len(self.lnc_manager.l1_active_lncs)}")
        except Exception as exc:
            print(f"[Monitor] error: {exc}")


# ---------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------

async def _main() -> None:
    app = CognitiveSystemApp()
    await app.bootstrap()

    try:
        await app.run()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(_main())