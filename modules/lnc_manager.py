#
# 文件: modules/lnc_manager.py
# 职责: LNC 管理器 (模块5) - AI 的“操作系统内核”(CPU)。
# 版本: V3 (集成 V1 创世逻辑与 V2 记忆固化，包含完整架构注释)
#
# [架构角色]:
# 1. 它是运行在 CPU 上的“主线程”管理者。
# 2. 它负责 LNC 的“三级缓存”生命周期管理：
#    - L3 (硬盘/Disk): 静态存储 ("潜意识晶体")
#    - L2 (内存/RAM): 预加载缓存 ("边缘意识")
#    - L1 (显存/VRAM): 激活运行 ("物理宇宙/意识流")
#
# [核心功能]:
# 1. 创世 (Genesis): 在启动时加载“硬编码”的归纳偏置 (V1)。
# 2. 加载 (Loading): 预测性地将 LNC 从 L3 搬运到 L1。
# 3. 固化 (Consolidation): 将经过物理演化(PhysicsKernel)的高价值 LNC 保存回硬盘。
#

import torch
import numpy as np
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

# 导入我们的核心模块
from . import neuron_schema as schema
from . import lnc_schema as lnc_schema
from .large_static_buffer import LargeStaticBuffer
# (我们将在下一份文件中定义 MessageBus，这里先做类型提示)
class_MessageBus = Any 


# --- 1. LNC 蓝图 (L3/L2 缓存的数据结构) ---

@dataclass
class LNCBlueprint:
    """
    LNC 蓝图 (LNC Blueprint)。
    
    这是存储在“硬盘”(L3)或“内存”(L2)中的 LNC“基因组”。
    它定义了 LNC 的初始状态和“归纳偏置” (Inductive Bias)。
    
    当一个 LNC 被“激活”到 LSB (L1) 时，它的这些属性会被
    memcpy (内存复制) 到 GPU 显存中。
    """
    lnc_id: str
    
    # "城市规划" 偏置: (W, H)，以 "地块" (Chunk) 为单位
    # 例如 (2, 2) 代表占用 64x64 像素的物理空间
    size_in_chunks: Tuple[int, int] = (1, 1) 
    
    # "贝叶斯" 偏置: 初始信念 (Prior Beliefs)
    # initial_k (精度): 越高代表这是“长期记忆”或“本能”，越难被改变。
    initial_k: float = 1.0
    
    # initial_theta_rate: 定义了伽马分布的形状
    initial_theta_rate: float = 1.0
    
    # "连接" 偏置: (V2 功能，V1 占位)
    # 定义它应该“订阅”哪些主题 (LNC-to-LNC 连接)
    subscribes_to: List[str] = field(default_factory=list)
    publishes_to: List[str] = field(default_factory=list)
    

# --- 2. LNC 管理器 (主类) ---

class LNCManager:
    """
    LNC 管理器 (LNCManager)。
    
    它不像 PhysicsKernel 那样在 GPU 上高频运行。
    它是一个 CPU 端的、事件驱动的“管家”。
    它负责“搬运物质”，让 GPU 专注于“演算物理”。
    """
    
    # 定义“城市规划”的“地块”大小 (单位: 像素/神经元)
    # LSB 是 2048x2048, 所以 (2048/32) = 64x64 个地块网格
    CHUNK_SIZE: int = 32
    
    # [V2 新增] 记忆固化阈值
    # 只有当 LNC 在 GPU 中演化后的平均 k 值 (信念强度) 超过此阈值时，
    # 它才会被视为“值得记忆”并写回硬盘。
    # (这是“代谢调节”可以动态改变的参数)
    MEMORY_CONSOLIDATION_THRESHOLD: float = 1.5 

    def __init__(self, lsb: LargeStaticBuffer, bus: class_MessageBus):
        print("LNCManager: Initializing (CPU)...")
        self.lsb = lsb
        self.bus = bus
        self.device = torch.device("cpu") # LNCManager 自身运行在 CPU 上

        # --- L1 缓存管理: (GPU VRAM) ---
        # "城市规划" 地图 (64x64 的 numpy 数组)
        # 0 = 空闲, >0 = 被占用的 LNC ID (Hash)
        self.allocation_map_shape = (
            lsb.height // self.CHUNK_SIZE, 
            lsb.width // self.CHUNK_SIZE
        )
        self.allocation_map = np.zeros(self.allocation_map_shape, dtype=np.int32)
        
        # 活跃 LNC 注册表: ID -> (x, y, w, h) 坐标
        self.l1_active_lncs: Dict[int, Tuple[int, int, int, int]] = {}
        
        # --- L2 缓存管理: (CPU RAM) ---
        # 预加载的 LNC 蓝图对象
        self.l2_preloaded_lncs: Dict[str, LNCBlueprint] = {}

        # --- L3 缓存管理: (Disk) ---
        # 模拟的硬盘 LNC 蓝图数据库
        self.l3_disk_blueprints: Dict[str, LNCBlueprint] = {}
        self.pending_consolidation: List[int] = []
        
        # 初始化 V1 的模拟硬盘数据
        self._init_l3_disk_cache()
        self.snapshot_history: Dict[int, List[Dict[str, Any]]] = {}
        self.blanket_hierarchy: Dict[int, Dict[str, Any]] = {}
        self.role_bindings: Dict[str, int] = {}


    def bind_role_to_lnc(self, role: str, lnc_id: str) -> None:
        lnc_id_int = hash(lnc_id) % 2**16 if isinstance(lnc_id, str) else int(lnc_id)
        self.role_bindings[role] = lnc_id_int

    def get_role_bound_lnc(self, role: str) -> Optional[str]:
        lnc_id_int = self.role_bindings.get(role)
        if lnc_id_int is None:
            return None
        for key in list(self.l2_preloaded_lncs.keys()) + list(self.l3_disk_blueprints.keys()):
            if (hash(key) % 2**16) == lnc_id_int:
                return key
        return str(lnc_id_int)

    def get_role_bound_region(self, role: str) -> Optional[Tuple[int, int, int, int]]:
        lnc_id_int = self.role_bindings.get(role)
        if lnc_id_int is None:
            return None
        region = self.l1_active_lncs.get(lnc_id_int)
        if region is None:
            return None
        cx, cy, cw, ch = region
        return (
            cx * self.CHUNK_SIZE,
            cy * self.CHUNK_SIZE,
            cw * self.CHUNK_SIZE,
            ch * self.CHUNK_SIZE,
        )

    def _init_l3_disk_cache(self) -> None:
        """
        (V1 模拟) "模拟"从硬盘读取 LNC 蓝图。
        这就是我们的“V1 归纳偏置库”。
        """
        print("LNCManager: Initializing L3 Disk Cache (V1 Simulation)...")
        
        # [归纳偏置 1]: 皮肤触觉处理单元
        # 一个 2x2 (64x64 像素) 的 LNC
        self.l3_disk_blueprints["LNC-V1-Skin"] = LNCBlueprint(
            lnc_id="LNC-V1-Skin",
            size_in_chunks=(2, 2), 
            initial_k=1.0,          # 初始信念较低，允许快速学习
            initial_theta_rate=1.0,
            subscribes_to=["SENSOR_SKIN"] # V2: 将用于自动连接逻辑
        )
        
        # [归纳偏置 2]: 自指/规划单元 (LNC-Planner)
        # 一个 1x1 (32x32 像素) 的 LNC
        self.l3_disk_blueprints["LNC-Planner"] = LNCBlueprint(
            lnc_id="LNC-Planner",
            size_in_chunks=(1, 1), 
            initial_k=5.0,          # 初始信念很高 (本能/固化逻辑)
            initial_theta_rate=0.5,
            subscribes_to=["LSB_METRICS"] # V2: 观测 LSB 状态 (元认知)
        )
        print(f"LNCManager: L3 Disk Cache populated with {len(self.l3_disk_blueprints)} blueprints.")

    def start_genesis(self, genesis_list: List[str]) -> None:
        """
        执行“创世” (Genesis) 程序。
        
        这是 AI 启动时的第一步。它负责将“V1 归纳偏置”列表中的 LNC
        从 L3 (硬盘) 加载到 L1 (显存)，并“激活”它们。
        如果没有这一步，GPU 上的 PhysicsKernel 将对着一片虚空空转。
        
        :param genesis_list: [硬编码] 要“创世”的 LNC ID 列表。
        """
        print("="*50)
        print("LNCManager: [GENESIS] 启动“创世”程序...")
        print(f"LNCManager: [GENESIS] V1 归纳偏置列表: {genesis_list}")
        
        for lnc_id in genesis_list:
            print(f"LNCManager: [GENESIS] 正在加载 {lnc_id}...")
            success = self._load_lnc_to_l1(lnc_id)
            if not success:
                print(f"!! LNCManager: [GENESIS] 灾难性错误 !!")
                print(f"!! 无法为 {lnc_id} 分配 L1 (VRAM) 地块。LSB 空间不足。")
                print(f"!! “创世”失败。")
                raise RuntimeError(f"Genesis failed: Could not allocate space for {lnc_id}")
        
        print("LNCManager: [GENESIS] “创世”程序完成。")
        print(f"LNCManager: L1 (VRAM) 地块占用图概览:\n{self.allocation_map[:10, :10]} ...")
        print("="*50)

    # --- 核心管道：加载 (Disk -> RAM -> GPU) ---
    
    def _load_lnc_to_l1(self, lnc_id: str) -> bool:
        """
        执行 L3 -> L2 -> L1 的完整“加载管道”。
        这是将“死数据”变为“活智能”的过程。
        """
        
        # 1. [L3 -> L2] (硬盘 -> 内存)
        blueprint = self._load_blueprint_from_disk(lnc_id)
        if blueprint is None:
            print(f"LNCManager Error: 蓝图 {lnc_id} 在 L3 (硬盘) 中未找到。")
            return False
        
        # 2. [L1 城市规划] (在 CPU 上计算)
        # 在 2048x2048 的 LSB 中找到一块空闲区域
        size_in_chunks = blueprint.size_in_chunks
        chunk_coord = self._find_free_chunk(size_in_chunks)
        
        if chunk_coord is None:
            print(f"LNCManager Error: L1 (VRAM) 空间不足。无法为 {lnc_id} (大小 {size_in_chunks}) 找到空闲地块。")
            # V2: 在这里应该触发“自然选择/卸载”逻辑来腾出空间
            return False
            
        chunk_x, chunk_y = chunk_coord
        chunk_w, chunk_h = size_in_chunks
        
        # 3. [L2 -> L1] (内存 -> 显存)
        print(f"LNCManager: ...L1 (VRAM) 地块已分配: at ({chunk_x}, {chunk_y}), size {size_in_chunks}")
        
        # 将“地块”坐标转换为“像素”坐标 (因为 LSB 使用像素索引)
        pixel_x = chunk_x * self.CHUNK_SIZE
        pixel_y = chunk_y * self.CHUNK_SIZE
        pixel_w = chunk_w * self.CHUNK_SIZE
        pixel_h = chunk_h * self.CHUNK_SIZE
        
        # 从 LSB 获取该区域的 GPU 张量“切片” (View)
        lnc_slice = self.lsb.get_slice(pixel_x, pixel_y, pixel_w, pixel_h)
        
        # 将“蓝图”(内存中的初始参数) 写入“切片”(显存)
        # 这是“创世”的核心 memcpy 操作
        self._write_genesis_data_to_slice(lnc_slice, blueprint)
        
        # 4. [注册] (更新 CPU 上的“缓存”记录)
        lnc_id_int = hash(blueprint.lnc_id) % 2**16 # 简单的字符串 -> 整数 ID
        # 更新占用图
        self.allocation_map[chunk_y:chunk_y+chunk_h, chunk_x:chunk_x+chunk_w] = lnc_id_int
        # 更新活跃表
        self.l1_active_lncs[lnc_id_int] = (chunk_x, chunk_y, chunk_w, chunk_h)
        
        print(f"LNCManager: ...{lnc_id} (ID: {lnc_id_int}) 已激活至 L1 (VRAM)。")
        return True

    # --- [V2 新增] 核心管道：保存/记忆固化 (GPU -> RAM -> Disk) ---
    
    def save_lnc_snapshot(self, lnc_id_int: int) -> bool:
        """
        执行 L1 -> L2 -> L3 的“记忆固化”管道。
        
        这是 AI 的“海马体”功能。它将 GPU 中经过“物理演算”和
        “神经递质强化”后的状态，选择性地保存回硬盘。
        
        :param lnc_id_int: 要保存的 LNC 的整数 ID。
        :return: bool, True 表示保存成功，False 表示被过滤掉(遗忘)。
        """
        if lnc_id_int not in self.l1_active_lncs:
            print(f"LNCManager: Cannot save unknown ID {lnc_id_int}")
            return False
            
        (cx, cy, cw, ch) = self.l1_active_lncs[lnc_id_int]
        
        # 1. [GPU -> CPU] 回读数据 (VRAM -> RAM)
        # 这是一个昂贵的操作 (Device -> Host copy)，所以只对特定的 LNC 做
        pixel_x, pixel_y = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pixel_w, pixel_h = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        
        # 获取 GPU 切片
        gpu_slice = self.lsb.get_slice(pixel_x, pixel_y, pixel_w, pixel_h)
        
        # 复制到 CPU (Tensor -> Tensor/Numpy)
        cpu_snapshot = gpu_slice.cpu() 
        
        # 2. [筛选] 检查记忆价值 (k 值)
        # 读取 BELIEF_SHAPE_K_INDEX
        # 如果 PhysicsKernel 中的贝叶斯更新增加了 k 值 (精度)，
        # 说明这个 LNC 经历了重要的学习。
        k_values = cpu_snapshot[..., schema.BELIEF_SHAPE_K_INDEX]
        avg_k = float(torch.mean(k_values))
        
        print(f"LNCManager: Evaluating LNC {lnc_id_int} for consolidation... Avg k={avg_k:.2f}")
        
        if avg_k < self.MEMORY_CONSOLIDATION_THRESHOLD:
            # [筛选/遗忘]: 记忆强度不足，丢弃。
            print(f"LNCManager: -> Memory weak (k < {self.MEMORY_CONSOLIDATION_THRESHOLD}). DISCARDING changes.")
            return False
            
        # 3. [更新] 更新 L2 蓝图
        # 我们需要找到它对应的 LNC ID 字符串 (这需要一个反向查找表，V1简化处理)
        # 假设我们能找到 blueprint 对象...
        target_blueprint = None
        for bp in self.l2_preloaded_lncs.values():
            if (hash(bp.lnc_id) % 2**16) == lnc_id_int:
                target_blueprint = bp
                break
        
        if target_blueprint:
            print(f"LNCManager: -> Memory strong! Consolidating to disk...")
            
            # 更新蓝图中的信念 (简化为平均值，V2应该是保存完整的权重矩阵)
            # 这里我们将 GPU 上学到的新参数，写回到蓝图对象中
            target_blueprint.initial_k = avg_k
            # 同时也保存学习到的 gamma rate
            target_blueprint.initial_theta_rate = float(torch.mean(cpu_snapshot[..., schema.BELIEF_RATE_THETA_INDEX]))
            
            # 4. [L2 -> L3] 写入硬盘
            # 更新“硬盘”数据库，这样下次加载时，它就是“变聪明”的版本了
            self.l3_disk_blueprints[target_blueprint.lnc_id] = target_blueprint
            print(f"LNCManager: Saved {target_blueprint.lnc_id} to Disk (L3).")
            if lnc_id_int in self.pending_consolidation:
                self.pending_consolidation.remove(lnc_id_int)
            return True
            
        return False

    def mark_pending_consolidation(self, lnc_id_int: int) -> None:
        """登记需要在 drift/sleep 窗口固化的 LNC。"""
        if lnc_id_int in self.l1_active_lncs and lnc_id_int not in self.pending_consolidation:
            self.pending_consolidation.append(lnc_id_int)

    def process_pending_consolidation(self, max_items: int = 2) -> int:
        """低频处理 pending 队列，避免 burst 时写盘抖动。"""
        done = 0
        for lnc_id_int in list(self.pending_consolidation)[:max_items]:
            if self.save_lnc_snapshot(lnc_id_int):
                done += 1
        return done

    def save_structured_snapshot(self, lnc_id_int: int, disk_manager: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """保存结构化快照（key/frame/residual_stats）并可写入 DiskManager 版本库。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        gpu_slice = self.lsb.get_slice(px, py, pw, ph)
        snap = gpu_slice.cpu()

        payload: Dict[str, Any] = {
            "lnc_id_int": lnc_id_int,
            "avg_k": float(torch.mean(snap[..., schema.BELIEF_SHAPE_K_INDEX])),
            "avg_theta": float(torch.mean(snap[..., schema.BELIEF_RATE_THETA_INDEX])),
            "residual_energy": float(torch.mean(torch.abs(snap[..., schema.RESIDUAL_ENERGY_INDEX]))),
            "residual_stats": {
                "trace_strength": float(torch.mean(snap[..., schema.TRACE_STRENGTH_INDEX])),
                "trace_decay": float(torch.mean(snap[..., schema.TRACE_DECAY_INDEX])),
            },
            "frame": {
                "alpha": 1.0,
                "phi": 0.0,
                "delta": 0.0,
                "sigma": 1.0,
            },
            "key_sig": [0.0 for _ in range(lnc_schema.LNC_KEY_SIG_DIM)],
        }

        self.snapshot_history.setdefault(lnc_id_int, []).append(payload)
        if disk_manager is not None:
            disk_manager.save_structured_snapshot(str(lnc_id_int), payload)
        return payload

    def rollback_structured_snapshot(self,
                                     lnc_id_int: int,
                                     disk_manager: Optional[Any] = None,
                                     target_version: Optional[int] = None) -> bool:
        """从结构化快照回滚（优先 DiskManager 版本库）。"""
        payload: Optional[Dict[str, Any]] = None
        if disk_manager is not None:
            payload = disk_manager.rollback_snapshot(str(lnc_id_int), target_version)

        if payload is None:
            history = self.snapshot_history.get(lnc_id_int, [])
            if len(history) < 1:
                return False
            payload = history[-2] if len(history) > 1 else history[-1]

        if lnc_id_int not in self.l1_active_lncs:
            return False

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        lnc_slice = self.lsb.get_slice(px, py, pw, ph)
        lnc_slice[..., schema.BELIEF_SHAPE_K_INDEX] = payload.get("avg_k", 1.0)
        lnc_slice[..., schema.BELIEF_RATE_THETA_INDEX] = payload.get("avg_theta", 1.0)
        return True

    def register_blanket_level(self, lnc_id_int: int, level: int, parent_id: Optional[int] = None) -> None:
        """登记多尺度 blanket 层级关系。"""
        node = self.blanket_hierarchy.setdefault(lnc_id_int, {"level": level, "parent": parent_id, "children": []})
        node["level"] = level
        node["parent"] = parent_id
        if parent_id is not None:
            parent = self.blanket_hierarchy.setdefault(parent_id, {"level": max(0, level - 1), "parent": None, "children": []})
            if lnc_id_int not in parent["children"]:
                parent["children"].append(lnc_id_int)

    def split_lnc_with_shadow(self, lnc_id_int: int, shadow_suffix: str = "S1") -> Optional[int]:
        """Phase4: 将 LNC 通过 shadow 分裂为一个新蓝图并尝试加载到 L1。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        source_bp: Optional[LNCBlueprint] = None
        for bp in self.l2_preloaded_lncs.values():
            if (hash(bp.lnc_id) % 2**16) == lnc_id_int:
                source_bp = bp
                break
        if source_bp is None:
            return None

        new_lnc_id = f"{source_bp.lnc_id}-shadow-{shadow_suffix}"
        new_bp = LNCBlueprint(
            lnc_id=new_lnc_id,
            size_in_chunks=source_bp.size_in_chunks,
            initial_k=max(0.5, source_bp.initial_k * 0.9),
            initial_theta_rate=source_bp.initial_theta_rate,
            subscribes_to=list(source_bp.subscribes_to),
            publishes_to=list(source_bp.publishes_to),
        )
        self.l3_disk_blueprints[new_lnc_id] = new_bp
        self.l2_preloaded_lncs[new_lnc_id] = new_bp

        if not self._load_lnc_to_l1(new_lnc_id):
            return None

        new_lnc_id_int = hash(new_lnc_id) % 2**16
        parent_level = self.blanket_hierarchy.get(lnc_id_int, {}).get("level", 0)
        self.register_blanket_level(new_lnc_id_int, level=parent_level + 1, parent_id=lnc_id_int)
        return new_lnc_id_int

    def merge_lncs(self, winner_lnc_id_int: int, loser_lnc_id_int: int) -> bool:
        """Phase4: 合并两个 LNC，保留 winner 并回收 loser。"""
        if winner_lnc_id_int not in self.l1_active_lncs or loser_lnc_id_int not in self.l1_active_lncs:
            return False

        # 先固化 winner，避免合并导致历史丢失。
        self.mark_pending_consolidation(winner_lnc_id_int)
        self.process_pending_consolidation(max_items=1)

        # 将 loser 的层级子节点挂到 winner。
        loser_node = self.blanket_hierarchy.get(loser_lnc_id_int)
        if loser_node:
            for child in loser_node.get("children", []):
                self.register_blanket_level(child, level=self.blanket_hierarchy.get(child, {}).get("level", 1), parent_id=winner_lnc_id_int)

        self.unload_lnc_from_l1(loser_lnc_id_int)
        self.blanket_hierarchy.pop(loser_lnc_id_int, None)
        return True

    def mark_pending_consolidation(self, lnc_id_int: int) -> None:
        """登记需要在 drift/sleep 窗口固化的 LNC。"""
        if lnc_id_int in self.l1_active_lncs and lnc_id_int not in self.pending_consolidation:
            self.pending_consolidation.append(lnc_id_int)

    def process_pending_consolidation(self, max_items: int = 2) -> int:
        """低频处理 pending 队列，避免 burst 时写盘抖动。"""
        done = 0
        for lnc_id_int in list(self.pending_consolidation)[:max_items]:
            if self.save_lnc_snapshot(lnc_id_int):
                done += 1
        return done

    def save_structured_snapshot(self, lnc_id_int: int, disk_manager: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """保存结构化快照（key/frame/residual_stats）并可写入 DiskManager 版本库。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        gpu_slice = self.lsb.get_slice(px, py, pw, ph)
        snap = gpu_slice.cpu()

        payload: Dict[str, Any] = {
            "lnc_id_int": lnc_id_int,
            "avg_k": float(torch.mean(snap[..., schema.BELIEF_SHAPE_K_INDEX])),
            "avg_theta": float(torch.mean(snap[..., schema.BELIEF_RATE_THETA_INDEX])),
            "residual_energy": float(torch.mean(torch.abs(snap[..., schema.RESIDUAL_ENERGY_INDEX]))),
            "residual_stats": {
                "trace_strength": float(torch.mean(snap[..., schema.TRACE_STRENGTH_INDEX])),
                "trace_decay": float(torch.mean(snap[..., schema.TRACE_DECAY_INDEX])),
            },
            "frame": {
                "alpha": 1.0,
                "phi": 0.0,
                "delta": 0.0,
                "sigma": 1.0,
            },
            "key_sig": [0.0 for _ in range(lnc_schema.LNC_KEY_SIG_DIM)],
        }

        self.snapshot_history.setdefault(lnc_id_int, []).append(payload)
        if disk_manager is not None:
            disk_manager.save_structured_snapshot(str(lnc_id_int), payload)
        return payload

    def rollback_structured_snapshot(self,
                                     lnc_id_int: int,
                                     disk_manager: Optional[Any] = None,
                                     target_version: Optional[int] = None) -> bool:
        """从结构化快照回滚（优先 DiskManager 版本库）。"""
        payload: Optional[Dict[str, Any]] = None
        if disk_manager is not None:
            payload = disk_manager.rollback_snapshot(str(lnc_id_int), target_version)

        if payload is None:
            history = self.snapshot_history.get(lnc_id_int, [])
            if len(history) < 1:
                return False
            payload = history[-2] if len(history) > 1 else history[-1]

        if lnc_id_int not in self.l1_active_lncs:
            return False

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        lnc_slice = self.lsb.get_slice(px, py, pw, ph)
        lnc_slice[..., schema.BELIEF_SHAPE_K_INDEX] = payload.get("avg_k", 1.0)
        lnc_slice[..., schema.BELIEF_RATE_THETA_INDEX] = payload.get("avg_theta", 1.0)
        return True

    def register_blanket_level(self, lnc_id_int: int, level: int, parent_id: Optional[int] = None) -> None:
        """登记多尺度 blanket 层级关系。"""
        node = self.blanket_hierarchy.setdefault(lnc_id_int, {"level": level, "parent": parent_id, "children": []})
        node["level"] = level
        node["parent"] = parent_id
        if parent_id is not None:
            parent = self.blanket_hierarchy.setdefault(parent_id, {"level": max(0, level - 1), "parent": None, "children": []})
            if lnc_id_int not in parent["children"]:
                parent["children"].append(lnc_id_int)

    def split_lnc_with_shadow(self, lnc_id_int: int, shadow_suffix: str = "S1") -> Optional[int]:
        """Phase4: 将 LNC 通过 shadow 分裂为一个新蓝图并尝试加载到 L1。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        source_bp: Optional[LNCBlueprint] = None
        for bp in self.l2_preloaded_lncs.values():
            if (hash(bp.lnc_id) % 2**16) == lnc_id_int:
                source_bp = bp
                break
        if source_bp is None:
            return None

        new_lnc_id = f"{source_bp.lnc_id}-shadow-{shadow_suffix}"
        new_bp = LNCBlueprint(
            lnc_id=new_lnc_id,
            size_in_chunks=source_bp.size_in_chunks,
            initial_k=max(0.5, source_bp.initial_k * 0.9),
            initial_theta_rate=source_bp.initial_theta_rate,
            subscribes_to=list(source_bp.subscribes_to),
            publishes_to=list(source_bp.publishes_to),
        )
        self.l3_disk_blueprints[new_lnc_id] = new_bp
        self.l2_preloaded_lncs[new_lnc_id] = new_bp

        if not self._load_lnc_to_l1(new_lnc_id):
            return None

        new_lnc_id_int = hash(new_lnc_id) % 2**16
        parent_level = self.blanket_hierarchy.get(lnc_id_int, {}).get("level", 0)
        self.register_blanket_level(new_lnc_id_int, level=parent_level + 1, parent_id=lnc_id_int)
        return new_lnc_id_int

    def merge_lncs(self, winner_lnc_id_int: int, loser_lnc_id_int: int) -> bool:
        """Phase4: 合并两个 LNC，保留 winner 并回收 loser。"""
        if winner_lnc_id_int not in self.l1_active_lncs or loser_lnc_id_int not in self.l1_active_lncs:
            return False

        # 先固化 winner，避免合并导致历史丢失。
        self.mark_pending_consolidation(winner_lnc_id_int)
        self.process_pending_consolidation(max_items=1)

        # 将 loser 的层级子节点挂到 winner。
        loser_node = self.blanket_hierarchy.get(loser_lnc_id_int)
        if loser_node:
            for child in loser_node.get("children", []):
                self.register_blanket_level(child, level=self.blanket_hierarchy.get(child, {}).get("level", 1), parent_id=winner_lnc_id_int)

        self.unload_lnc_from_l1(loser_lnc_id_int)
        self.blanket_hierarchy.pop(loser_lnc_id_int, None)
        return True

    def mark_pending_consolidation(self, lnc_id_int: int) -> None:
        """登记需要在 drift/sleep 窗口固化的 LNC。"""
        if lnc_id_int in self.l1_active_lncs and lnc_id_int not in self.pending_consolidation:
            self.pending_consolidation.append(lnc_id_int)

    def process_pending_consolidation(self, max_items: int = 2) -> int:
        """低频处理 pending 队列，避免 burst 时写盘抖动。"""
        done = 0
        for lnc_id_int in list(self.pending_consolidation)[:max_items]:
            if self.save_lnc_snapshot(lnc_id_int):
                done += 1
        return done

    def save_structured_snapshot(self, lnc_id_int: int, disk_manager: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """保存结构化快照（key/frame/residual_stats）并可写入 DiskManager 版本库。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        gpu_slice = self.lsb.get_slice(px, py, pw, ph)
        snap = gpu_slice.cpu()

        payload: Dict[str, Any] = {
            "lnc_id_int": lnc_id_int,
            "avg_k": float(torch.mean(snap[..., schema.BELIEF_SHAPE_K_INDEX])),
            "avg_theta": float(torch.mean(snap[..., schema.BELIEF_RATE_THETA_INDEX])),
            "residual_energy": float(torch.mean(torch.abs(snap[..., schema.RESIDUAL_ENERGY_INDEX]))),
            "residual_stats": {
                "trace_strength": float(torch.mean(snap[..., schema.TRACE_STRENGTH_INDEX])),
                "trace_decay": float(torch.mean(snap[..., schema.TRACE_DECAY_INDEX])),
            },
            "frame": {
                "alpha": 1.0,
                "phi": 0.0,
                "delta": 0.0,
                "sigma": 1.0,
            },
            "key_sig": [0.0 for _ in range(lnc_schema.LNC_KEY_SIG_DIM)],
        }

        self.snapshot_history.setdefault(lnc_id_int, []).append(payload)
        if disk_manager is not None:
            disk_manager.save_structured_snapshot(str(lnc_id_int), payload)
        return payload

    def rollback_structured_snapshot(self,
                                     lnc_id_int: int,
                                     disk_manager: Optional[Any] = None,
                                     target_version: Optional[int] = None) -> bool:
        """从结构化快照回滚（优先 DiskManager 版本库）。"""
        payload: Optional[Dict[str, Any]] = None
        if disk_manager is not None:
            payload = disk_manager.rollback_snapshot(str(lnc_id_int), target_version)

        if payload is None:
            history = self.snapshot_history.get(lnc_id_int, [])
            if len(history) < 1:
                return False
            payload = history[-2] if len(history) > 1 else history[-1]

        if lnc_id_int not in self.l1_active_lncs:
            return False

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        lnc_slice = self.lsb.get_slice(px, py, pw, ph)
        lnc_slice[..., schema.BELIEF_SHAPE_K_INDEX] = payload.get("avg_k", 1.0)
        lnc_slice[..., schema.BELIEF_RATE_THETA_INDEX] = payload.get("avg_theta", 1.0)
        return True

    def register_blanket_level(self, lnc_id_int: int, level: int, parent_id: Optional[int] = None) -> None:
        """登记多尺度 blanket 层级关系。"""
        node = self.blanket_hierarchy.setdefault(lnc_id_int, {"level": level, "parent": parent_id, "children": []})
        node["level"] = level
        node["parent"] = parent_id
        if parent_id is not None:
            parent = self.blanket_hierarchy.setdefault(parent_id, {"level": max(0, level - 1), "parent": None, "children": []})
            if lnc_id_int not in parent["children"]:
                parent["children"].append(lnc_id_int)

    def split_lnc_with_shadow(self, lnc_id_int: int, shadow_suffix: str = "S1") -> Optional[int]:
        """Phase4: 将 LNC 通过 shadow 分裂为一个新蓝图并尝试加载到 L1。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        source_bp: Optional[LNCBlueprint] = None
        for bp in self.l2_preloaded_lncs.values():
            if (hash(bp.lnc_id) % 2**16) == lnc_id_int:
                source_bp = bp
                break
        if source_bp is None:
            return None

        new_lnc_id = f"{source_bp.lnc_id}-shadow-{shadow_suffix}"
        new_bp = LNCBlueprint(
            lnc_id=new_lnc_id,
            size_in_chunks=source_bp.size_in_chunks,
            initial_k=max(0.5, source_bp.initial_k * 0.9),
            initial_theta_rate=source_bp.initial_theta_rate,
            subscribes_to=list(source_bp.subscribes_to),
            publishes_to=list(source_bp.publishes_to),
        )
        self.l3_disk_blueprints[new_lnc_id] = new_bp
        self.l2_preloaded_lncs[new_lnc_id] = new_bp

        if not self._load_lnc_to_l1(new_lnc_id):
            return None

        new_lnc_id_int = hash(new_lnc_id) % 2**16
        parent_level = self.blanket_hierarchy.get(lnc_id_int, {}).get("level", 0)
        self.register_blanket_level(new_lnc_id_int, level=parent_level + 1, parent_id=lnc_id_int)
        return new_lnc_id_int

    def merge_lncs(self, winner_lnc_id_int: int, loser_lnc_id_int: int) -> bool:
        """Phase4: 合并两个 LNC，保留 winner 并回收 loser。"""
        if winner_lnc_id_int not in self.l1_active_lncs or loser_lnc_id_int not in self.l1_active_lncs:
            return False

        # 先固化 winner，避免合并导致历史丢失。
        self.mark_pending_consolidation(winner_lnc_id_int)
        self.process_pending_consolidation(max_items=1)

        # 将 loser 的层级子节点挂到 winner。
        loser_node = self.blanket_hierarchy.get(loser_lnc_id_int)
        if loser_node:
            for child in loser_node.get("children", []):
                self.register_blanket_level(child, level=self.blanket_hierarchy.get(child, {}).get("level", 1), parent_id=winner_lnc_id_int)

        self.unload_lnc_from_l1(loser_lnc_id_int)
        self.blanket_hierarchy.pop(loser_lnc_id_int, None)
        return True

    def mark_pending_consolidation(self, lnc_id_int: int) -> None:
        """登记需要在 drift/sleep 窗口固化的 LNC。"""
        if lnc_id_int in self.l1_active_lncs and lnc_id_int not in self.pending_consolidation:
            self.pending_consolidation.append(lnc_id_int)

    def process_pending_consolidation(self, max_items: int = 2) -> int:
        """低频处理 pending 队列，避免 burst 时写盘抖动。"""
        done = 0
        for lnc_id_int in list(self.pending_consolidation)[:max_items]:
            if self.save_lnc_snapshot(lnc_id_int):
                done += 1
        return done

    def save_structured_snapshot(self, lnc_id_int: int, disk_manager: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """保存结构化快照（key/frame/residual_stats）并可写入 DiskManager 版本库。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        gpu_slice = self.lsb.get_slice(px, py, pw, ph)
        snap = gpu_slice.cpu()

        payload: Dict[str, Any] = {
            "lnc_id_int": lnc_id_int,
            "avg_k": float(torch.mean(snap[..., schema.BELIEF_SHAPE_K_INDEX])),
            "avg_theta": float(torch.mean(snap[..., schema.BELIEF_RATE_THETA_INDEX])),
            "residual_energy": float(torch.mean(torch.abs(snap[..., schema.RESIDUAL_ENERGY_INDEX]))),
            "residual_stats": {
                "trace_strength": float(torch.mean(snap[..., schema.TRACE_STRENGTH_INDEX])),
                "trace_decay": float(torch.mean(snap[..., schema.TRACE_DECAY_INDEX])),
            },
            "frame": {
                "alpha": 1.0,
                "phi": 0.0,
                "delta": 0.0,
                "sigma": 1.0,
            },
            "key_sig": [0.0 for _ in range(lnc_schema.LNC_KEY_SIG_DIM)],
        }

        self.snapshot_history.setdefault(lnc_id_int, []).append(payload)
        if disk_manager is not None:
            disk_manager.save_structured_snapshot(str(lnc_id_int), payload)
        return payload

    def rollback_structured_snapshot(self,
                                     lnc_id_int: int,
                                     disk_manager: Optional[Any] = None,
                                     target_version: Optional[int] = None) -> bool:
        """从结构化快照回滚（优先 DiskManager 版本库）。"""
        payload: Optional[Dict[str, Any]] = None
        if disk_manager is not None:
            payload = disk_manager.rollback_snapshot(str(lnc_id_int), target_version)

        if payload is None:
            history = self.snapshot_history.get(lnc_id_int, [])
            if len(history) < 1:
                return False
            payload = history[-2] if len(history) > 1 else history[-1]

        if lnc_id_int not in self.l1_active_lncs:
            return False

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        lnc_slice = self.lsb.get_slice(px, py, pw, ph)
        lnc_slice[..., schema.BELIEF_SHAPE_K_INDEX] = payload.get("avg_k", 1.0)
        lnc_slice[..., schema.BELIEF_RATE_THETA_INDEX] = payload.get("avg_theta", 1.0)
        return True

    def register_blanket_level(self, lnc_id_int: int, level: int, parent_id: Optional[int] = None) -> None:
        """登记多尺度 blanket 层级关系。"""
        node = self.blanket_hierarchy.setdefault(lnc_id_int, {"level": level, "parent": parent_id, "children": []})
        node["level"] = level
        node["parent"] = parent_id
        if parent_id is not None:
            parent = self.blanket_hierarchy.setdefault(parent_id, {"level": max(0, level - 1), "parent": None, "children": []})
            if lnc_id_int not in parent["children"]:
                parent["children"].append(lnc_id_int)

    def split_lnc_with_shadow(self, lnc_id_int: int, shadow_suffix: str = "S1") -> Optional[int]:
        """Phase4: 将 LNC 通过 shadow 分裂为一个新蓝图并尝试加载到 L1。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        source_bp: Optional[LNCBlueprint] = None
        for bp in self.l2_preloaded_lncs.values():
            if (hash(bp.lnc_id) % 2**16) == lnc_id_int:
                source_bp = bp
                break
        if source_bp is None:
            return None

        new_lnc_id = f"{source_bp.lnc_id}-shadow-{shadow_suffix}"
        new_bp = LNCBlueprint(
            lnc_id=new_lnc_id,
            size_in_chunks=source_bp.size_in_chunks,
            initial_k=max(0.5, source_bp.initial_k * 0.9),
            initial_theta_rate=source_bp.initial_theta_rate,
            subscribes_to=list(source_bp.subscribes_to),
            publishes_to=list(source_bp.publishes_to),
        )
        self.l3_disk_blueprints[new_lnc_id] = new_bp
        self.l2_preloaded_lncs[new_lnc_id] = new_bp

        if not self._load_lnc_to_l1(new_lnc_id):
            return None

        new_lnc_id_int = hash(new_lnc_id) % 2**16
        parent_level = self.blanket_hierarchy.get(lnc_id_int, {}).get("level", 0)
        self.register_blanket_level(new_lnc_id_int, level=parent_level + 1, parent_id=lnc_id_int)
        return new_lnc_id_int

    def merge_lncs(self, winner_lnc_id_int: int, loser_lnc_id_int: int) -> bool:
        """Phase4: 合并两个 LNC，保留 winner 并回收 loser。"""
        if winner_lnc_id_int not in self.l1_active_lncs or loser_lnc_id_int not in self.l1_active_lncs:
            return False

        # 先固化 winner，避免合并导致历史丢失。
        self.mark_pending_consolidation(winner_lnc_id_int)
        self.process_pending_consolidation(max_items=1)

        # 将 loser 的层级子节点挂到 winner。
        loser_node = self.blanket_hierarchy.get(loser_lnc_id_int)
        if loser_node:
            for child in loser_node.get("children", []):
                self.register_blanket_level(child, level=self.blanket_hierarchy.get(child, {}).get("level", 1), parent_id=winner_lnc_id_int)

        self.unload_lnc_from_l1(loser_lnc_id_int)
        self.blanket_hierarchy.pop(loser_lnc_id_int, None)
        return True

    def mark_pending_consolidation(self, lnc_id_int: int) -> None:
        """登记需要在 drift/sleep 窗口固化的 LNC。"""
        if lnc_id_int in self.l1_active_lncs and lnc_id_int not in self.pending_consolidation:
            self.pending_consolidation.append(lnc_id_int)

    def process_pending_consolidation(self, max_items: int = 2) -> int:
        """低频处理 pending 队列，避免 burst 时写盘抖动。"""
        done = 0
        for lnc_id_int in list(self.pending_consolidation)[:max_items]:
            if self.save_lnc_snapshot(lnc_id_int):
                done += 1
        return done

    def save_structured_snapshot(self, lnc_id_int: int, disk_manager: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """保存结构化快照（key/frame/residual_stats）并可写入 DiskManager 版本库。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        gpu_slice = self.lsb.get_slice(px, py, pw, ph)
        snap = gpu_slice.cpu()

        payload: Dict[str, Any] = {
            "lnc_id_int": lnc_id_int,
            "avg_k": float(torch.mean(snap[..., schema.BELIEF_SHAPE_K_INDEX])),
            "avg_theta": float(torch.mean(snap[..., schema.BELIEF_RATE_THETA_INDEX])),
            "residual_energy": float(torch.mean(torch.abs(snap[..., schema.RESIDUAL_ENERGY_INDEX]))),
            "residual_stats": {
                "trace_strength": float(torch.mean(snap[..., schema.TRACE_STRENGTH_INDEX])),
                "trace_decay": float(torch.mean(snap[..., schema.TRACE_DECAY_INDEX])),
            },
            "frame": {
                "alpha": 1.0,
                "phi": 0.0,
                "delta": 0.0,
                "sigma": 1.0,
            },
            "key_sig": [0.0 for _ in range(lnc_schema.LNC_KEY_SIG_DIM)],
        }

        self.snapshot_history.setdefault(lnc_id_int, []).append(payload)
        if disk_manager is not None:
            disk_manager.save_structured_snapshot(str(lnc_id_int), payload)
        return payload

    def rollback_structured_snapshot(self,
                                     lnc_id_int: int,
                                     disk_manager: Optional[Any] = None,
                                     target_version: Optional[int] = None) -> bool:
        """从结构化快照回滚（优先 DiskManager 版本库）。"""
        payload: Optional[Dict[str, Any]] = None
        if disk_manager is not None:
            payload = disk_manager.rollback_snapshot(str(lnc_id_int), target_version)

        if payload is None:
            history = self.snapshot_history.get(lnc_id_int, [])
            if len(history) < 1:
                return False
            payload = history[-2] if len(history) > 1 else history[-1]

        if lnc_id_int not in self.l1_active_lncs:
            return False

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        lnc_slice = self.lsb.get_slice(px, py, pw, ph)
        lnc_slice[..., schema.BELIEF_SHAPE_K_INDEX] = payload.get("avg_k", 1.0)
        lnc_slice[..., schema.BELIEF_RATE_THETA_INDEX] = payload.get("avg_theta", 1.0)
        return True

    def register_blanket_level(self, lnc_id_int: int, level: int, parent_id: Optional[int] = None) -> None:
        """登记多尺度 blanket 层级关系。"""
        node = self.blanket_hierarchy.setdefault(lnc_id_int, {"level": level, "parent": parent_id, "children": []})
        node["level"] = level
        node["parent"] = parent_id
        if parent_id is not None:
            parent = self.blanket_hierarchy.setdefault(parent_id, {"level": max(0, level - 1), "parent": None, "children": []})
            if lnc_id_int not in parent["children"]:
                parent["children"].append(lnc_id_int)

    def split_lnc_with_shadow(self, lnc_id_int: int, shadow_suffix: str = "S1") -> Optional[int]:
        """Phase4: 将 LNC 通过 shadow 分裂为一个新蓝图并尝试加载到 L1。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        source_bp: Optional[LNCBlueprint] = None
        for bp in self.l2_preloaded_lncs.values():
            if (hash(bp.lnc_id) % 2**16) == lnc_id_int:
                source_bp = bp
                break
        if source_bp is None:
            return None

        new_lnc_id = f"{source_bp.lnc_id}-shadow-{shadow_suffix}"
        new_bp = LNCBlueprint(
            lnc_id=new_lnc_id,
            size_in_chunks=source_bp.size_in_chunks,
            initial_k=max(0.5, source_bp.initial_k * 0.9),
            initial_theta_rate=source_bp.initial_theta_rate,
            subscribes_to=list(source_bp.subscribes_to),
            publishes_to=list(source_bp.publishes_to),
        )
        self.l3_disk_blueprints[new_lnc_id] = new_bp
        self.l2_preloaded_lncs[new_lnc_id] = new_bp

        if not self._load_lnc_to_l1(new_lnc_id):
            return None

        new_lnc_id_int = hash(new_lnc_id) % 2**16
        parent_level = self.blanket_hierarchy.get(lnc_id_int, {}).get("level", 0)
        self.register_blanket_level(new_lnc_id_int, level=parent_level + 1, parent_id=lnc_id_int)
        return new_lnc_id_int

    def merge_lncs(self, winner_lnc_id_int: int, loser_lnc_id_int: int) -> bool:
        """Phase4: 合并两个 LNC，保留 winner 并回收 loser。"""
        if winner_lnc_id_int not in self.l1_active_lncs or loser_lnc_id_int not in self.l1_active_lncs:
            return False

        # 先固化 winner，避免合并导致历史丢失。
        self.mark_pending_consolidation(winner_lnc_id_int)
        self.process_pending_consolidation(max_items=1)

        # 将 loser 的层级子节点挂到 winner。
        loser_node = self.blanket_hierarchy.get(loser_lnc_id_int)
        if loser_node:
            for child in loser_node.get("children", []):
                self.register_blanket_level(child, level=self.blanket_hierarchy.get(child, {}).get("level", 1), parent_id=winner_lnc_id_int)

        self.unload_lnc_from_l1(loser_lnc_id_int)
        self.blanket_hierarchy.pop(loser_lnc_id_int, None)
        return True

    def mark_pending_consolidation(self, lnc_id_int: int) -> None:
        """登记需要在 drift/sleep 窗口固化的 LNC。"""
        if lnc_id_int in self.l1_active_lncs and lnc_id_int not in self.pending_consolidation:
            self.pending_consolidation.append(lnc_id_int)

    def process_pending_consolidation(self, max_items: int = 2) -> int:
        """低频处理 pending 队列，避免 burst 时写盘抖动。"""
        done = 0
        for lnc_id_int in list(self.pending_consolidation)[:max_items]:
            if self.save_lnc_snapshot(lnc_id_int):
                done += 1
        return done

    def save_structured_snapshot(self, lnc_id_int: int, disk_manager: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """保存结构化快照（key/frame/residual_stats）并可写入 DiskManager 版本库。"""
        if lnc_id_int not in self.l1_active_lncs:
            return None

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        gpu_slice = self.lsb.get_slice(px, py, pw, ph)
        snap = gpu_slice.cpu()

        payload: Dict[str, Any] = {
            "lnc_id_int": lnc_id_int,
            "avg_k": float(torch.mean(snap[..., schema.BELIEF_SHAPE_K_INDEX])),
            "avg_theta": float(torch.mean(snap[..., schema.BELIEF_RATE_THETA_INDEX])),
            "residual_energy": float(torch.mean(torch.abs(snap[..., schema.RESIDUAL_ENERGY_INDEX]))),
            "residual_stats": {
                "trace_strength": float(torch.mean(snap[..., schema.TRACE_STRENGTH_INDEX])),
                "trace_decay": float(torch.mean(snap[..., schema.TRACE_DECAY_INDEX])),
            },
            "frame": {
                "alpha": 1.0,
                "phi": 0.0,
                "delta": 0.0,
                "sigma": 1.0,
            },
            "key_sig": [0.0 for _ in range(lnc_schema.LNC_KEY_SIG_DIM)],
        }

        self.snapshot_history.setdefault(lnc_id_int, []).append(payload)
        if disk_manager is not None:
            disk_manager.save_structured_snapshot(str(lnc_id_int), payload)
        return payload

    def rollback_structured_snapshot(self,
                                     lnc_id_int: int,
                                     disk_manager: Optional[Any] = None,
                                     target_version: Optional[int] = None) -> bool:
        """从结构化快照回滚（优先 DiskManager 版本库）。"""
        payload: Optional[Dict[str, Any]] = None
        if disk_manager is not None:
            payload = disk_manager.rollback_snapshot(str(lnc_id_int), target_version)

        if payload is None:
            history = self.snapshot_history.get(lnc_id_int, [])
            if len(history) < 1:
                return False
            payload = history[-2] if len(history) > 1 else history[-1]

        if lnc_id_int not in self.l1_active_lncs:
            return False

        cx, cy, cw, ch = self.l1_active_lncs[lnc_id_int]
        px, py = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        pw, ph = cw * self.CHUNK_SIZE, ch * self.CHUNK_SIZE
        lnc_slice = self.lsb.get_slice(px, py, pw, ph)
        lnc_slice[..., schema.BELIEF_SHAPE_K_INDEX] = payload.get("avg_k", 1.0)
        lnc_slice[..., schema.BELIEF_RATE_THETA_INDEX] = payload.get("avg_theta", 1.0)
        return True

    # --- 辅助方法 ---
    
    def _load_blueprint_from_disk(self, lnc_id: str) -> Optional[LNCBlueprint]:
        """(V1 模拟) 检查 L3，如果存在则加载到 L2。"""
        if lnc_id not in self.l3_disk_blueprints:
            return None
        
        # (V1 模拟) 检查 L2 缓存
        if lnc_id not in self.l2_preloaded_lncs:
            print(f"LNCManager: ...模拟 L3->L2 (硬盘->内存) I/O ... {lnc_id}")
            self.l2_preloaded_lncs[lnc_id] = self.l3_disk_blueprints[lnc_id]
        
        return self.l2_preloaded_lncs[lnc_id]

    def _find_free_chunk(self, size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        (V1 城市规划) First-Fit 算法。
        在 CPU 上的 self.allocation_map 中查找空闲地块。
        """
        w, h = size
        map_h, map_w = self.allocation_map.shape
        
        for y in range(map_h - h + 1):
            for x in range(map_w - w + 1):
                # 检查 (x, y) 处的 (w, h) 矩形区域
                region = self.allocation_map[y:y+h, x:x+w]
                if np.all(region == 0): # 如果区域内所有值都为 0 (空闲)
                    return (x, y) # 找到！
        
        return None # 未找到

    def _write_genesis_data_to_slice(self, lnc_slice: torch.Tensor, blueprint: LNCBlueprint) -> None:
        """
        (V1 创世/激活) L2 -> L1, 内存 -> 显存。
        这是“归纳偏置”被“写入”物理宇宙的地方。
        """
        lnc_id_int = hash(blueprint.lnc_id) % 2**16
        
        # 1. [关键] 激活门控 (IS_ACTIVE)
        # 只有这里被设为 1.0，PhysicsKernel 才会开始计算
        lnc_slice[..., schema.IS_ACTIVE_INDEX] = 1.0
        
        # 2. 标记领土 (LNC_ID)
        lnc_slice[..., schema.LNC_ID_INDEX] = float(lnc_id_int)
        
        # 3. 写入“归纳偏置” / “学到的知识” (初始信念)
        lnc_slice[..., schema.BELIEF_SHAPE_K_INDEX] = blueprint.initial_k
        lnc_slice[..., schema.BELIEF_RATE_THETA_INDEX] = blueprint.initial_theta_rate
        
        # 4. 给予初始能量 (否则下一帧就会饿死)
        lnc_slice[..., schema.ENERGY_COST_INDEX] = 1.0
        
        # 5. (V1) 清理瞬时状态 (振幅/相位)
        # 这些状态应该由 PhysicsKernel 从 0 开始演化
        lnc_slice[..., schema.AMPLITUDE_INDEX] = 0.0
        lnc_slice[..., schema.PHASE_INDEX] = 0.0
        lnc_slice[..., schema.AMPLITUDE_PREV_INDEX] = 0.0
        lnc_slice[..., schema.PHASE_PREV_INDEX] = 0.0
        
        print(f"LNCManager: ...L2->L1 (内存->显存) 写入完成。{blueprint.lnc_id} 已在 LSB 中“存活”。")

    # (V2 占位符: 预测性预加载监听器)
    def start_listeners(self) -> asyncio.Task:
        async def _preload_listener_stub():
            print("LNCManager: V2 预测性预加载 (Preload) 监听器已启动 (V1 Stub)。")
            while True:
                await asyncio.sleep(60) 
        return asyncio.create_task(_preload_listener_stub())

    # (V2 占位符: 卸载逻辑)
    def unload_lnc_from_l1(self, lnc_id_int: int) -> None:
        """
        (V2 自然选择) L1 -> L2, 显存 -> 内存。
        “卸载”一个 LNC，释放其 LSB 地块。
        (在此之前通常会调用 save_lnc_snapshot)
        """
        print(f"LNCManager: [V2 STUB] 正在卸载 LNC ID {lnc_id_int}...")
        if lnc_id_int not in self.l1_active_lncs:
            return
            
        (chunk_x, chunk_y, chunk_w, chunk_h) = self.l1_active_lncs[lnc_id_int]
        
        # “杀死” LNC (在 LSB 上)
        # 将其标记为“非激活”
        pixel_x, pixel_y = chunk_x * self.CHUNK_SIZE, chunk_y * self.CHUNK_SIZE
        pixel_w, pixel_h = chunk_w * self.CHUNK_SIZE, chunk_h * self.CHUNK_SIZE
        
        lnc_slice = self.lsb.get_slice(pixel_x, pixel_y, pixel_w, pixel_h)
        lnc_slice[..., schema.IS_ACTIVE_INDEX] = 0.0
        lnc_slice[..., schema.LNC_ID_INDEX] = 0.0
        
        # 释放“地块”(在 CPU 上)
        self.allocation_map[chunk_y:chunk_y+chunk_h, chunk_x:chunk_x+chunk_w] = 0
        del self.l1_active_lncs[lnc_id_int]
        
        print(f"LNCManager: LNC ID {lnc_id_int} 已从 L1 (VRAM) 卸载。地块已释放。")





    def emit_proposal(self,
        lsb_tensor: torch.Tensor,
        region_slice: Tuple[slice, slice],
        topdown: float = 0.0,
        precision_delta: float = 0.0,
        phase_proposal: float = 0.0,
        decay: float = 0.95) -> None:
        """
        向共享提案缓冲沉积建议（不直接覆盖状态）。
        region_slice 形如 (slice(y0,y1), slice(x0,x1))
        """
        ys, xs = region_slice
        lsb_tensor[ys, xs, schema.PROPOSAL_TOP_DOWN_INDEX] += topdown
        lsb_tensor[ys, xs, schema.PROPOSAL_PRECISION_DELTA_INDEX] += precision_delta
        lsb_tensor[ys, xs, schema.PROPOSAL_PHASE_PROPOSAL_INDEX] += phase_proposal
        lsb_tensor[ys, xs, schema.PROPOSAL_DECAY_INDEX] = decay
       
