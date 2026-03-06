#
# 文件: modules/neurotransmitter_manager.py
# 职责: 提供对 LSB 中“化学波场”的控制接口。
#
# [V2 重构 - 波函数化学场]:
# - 废弃了全局标量字典。
# - 实现了“波函数发射器”模式。
# - 这是一个 CPU 端的“遥控器”，允许自指模块(LNC-Planner)
#   向 GPU 的 LSB 指定区域“注入”化学波 (Amplitude, Phase)。
#

import torch
import numpy as np
from typing import Tuple, Optional
from . import neuron_schema as schema

class NeurotransmitterManager:
    """
    神经递质管理器 (CPU -> GPU Interface)。
    
    它不存储化学状态（状态现在作为物理场存储在 GPU 的 LSB 中）。
    它负责生成“化学注入指令”，直接修改 LSB 张量。
    """

    # --- 定义相位语义 (Phase Semantics) ---
    # 使用弧度表示“情感/功能色彩” (Valence)
    
    PHASE_REWARD: float = 0.0              # 多巴胺 (Dopamine) -> 奖励/确认
    PHASE_PUNISHMENT: float = np.pi        # 血清素 (Serotonin) -> 惩罚/抑制
    PHASE_NOVELTY: float = np.pi / 2       # 去甲肾上腺素 (Norepinephrine) -> 惊异/扩展
    PHASE_CALM: float = -np.pi / 2         # 乙酰胆碱 (Acetylcholine) -> 专注/收敛

    def __init__(self):
        print("NeurotransmitterManager: Initializing (Wave Emitter)...")
        # V1: 我们暂时不需要复杂的内部状态
        pass

    def inject_chemical_wave(self, 
                             lsb_tensor: torch.Tensor, 
                             center: Tuple[int, int], 
                             radius: int, 
                             amplitude: float, 
                             phase: float) -> None:
        """
        (同步/即时) 向 LSB 的特定区域注入化学波。
        
        这通常由“自指 LNC”调用，用于施加注意力或情绪调控。
        
        :param lsb_tensor: 目标 LSB 张量 (在 GPU 上)
        :param center: 注入中心 (x, y)
        :param radius: 注入半径 (影响范围)
        :param amplitude: 强度 (Salience/Anchoring) - 对应精度权重
        :param phase: 类型 (Valence) - 对应情绪色彩
        """
        
        # 获取 LSB 的维度
        H, W, _ = lsb_tensor.shape
        cx, cy = center
        
        # 计算受影响的切片边界 (简单的方形包围盒，V2可用高斯核)
        x_min = max(0, cx - radius)
        x_max = min(W, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(H, cy + radius)
        
        # 获取切片引用 (View)
        target_slice = lsb_tensor[y_min:y_max, x_min:x_max]
        
        # --- 注入波函数 ---
        # 物理含义：我们将新的波“叠加”到现有的化学场上。
        
        # 1. 注入振幅 (Salience / Precision)
        # 我们直接增加该区域的“化学浓度”。
        # 无论相位如何，振幅总是累加的（显显著性增加）。
        target_slice[..., schema.NT_AMPLITUDE_INDEX] += amplitude
        
        # 2. 注入相位 (Valence)
        # V1 简化策略：由于相位叠加涉及到复数运算，
        # 在 V1 中我们采用“强信号主导”策略。
        # 如果注入的波很强 (amplitude > 0.5)，它将覆盖当前的相位。
        # (在 V2 中，PhysicsKernel 应处理真正的复数波干涉)
        
        if amplitude > 0.1: # 只有足够强的波才能改变情绪色彩
            # 创建一个掩码：只在注入波强度大于现有波强度的地方更新相位
            # (这里简化为直接覆盖，模拟强烈的神经递质释放)
            target_slice[..., schema.NT_PHASE_INDEX] = phase

    def broadcast_global_signal(self, 
                                lsb_tensor: torch.Tensor, 
                                amplitude: float, 
                                phase: float) -> None:
        """
        (全局广播) 向整个宇宙注入化学波。
        模拟“脑干”级别的全局调节 (如觉醒、睡眠、极度惊恐)。
        """
        # 全局增加振幅 (提升全脑的可塑性/锚定)
        lsb_tensor[..., schema.NT_AMPLITUDE_INDEX] += amplitude
        
        # 全局重置相位 (强行同步全脑的情绪状态)
        lsb_tensor[..., schema.NT_PHASE_INDEX] = phase
        
        # print(f"NTManager: 全局广播 [Amp={amplitude:.2f}, Phase={phase:.2f}]")

    def decay_chemicals(self, lsb_tensor: torch.Tensor, decay_rate: float = 0.95) -> None:
        """
        (可选 - CPU 辅助) 全局化学衰减。
        通常这个逻辑应该在 PhysicsKernel (GPU) 中运行，
        但在 V1 中，如果 Kernel 压力大，也可以由 Manager 在每帧调用一次。
        """
        lsb_tensor[..., schema.NT_AMPLITUDE_INDEX] *= decay_rate
        # 相位通常不衰减，而是保持直到被新波覆盖
