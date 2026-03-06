#
# 文件: modules/physics_kernel_v1.py
# 职责: 实现“物理定律”的V1版本 (最终版)。
#       - 实现了 IPhysicsKernel 接口。
#       - 集成了“振荡器”、“预测编码”、“精度加权(神经递质)”和“时觉”定律。
#

import torch
import math

# 导入“合约”
from .physics_kernel_base import IPhysicsKernel
from . import neuron_schema as schema
from typing import Optional, Tuple

class PhysicsKernelV1(IPhysicsKernel):
    """
    物理引擎 V1 (PhysicsKernelV1)。
    
    这是我们“AI宇宙”的物理定律实现。
    它在单个 GPU Kernel 中并行执行所有 LNC 的动力学演化。
    """
    
    def __init__(self, 
                 k_coupling_const: float = 0.1, 
                 energy_replenish_rate: float = 0.01,
                 base_learning_rate: float = 0.01,
                 chemical_decay_rate: float = 0.95):
        """
        初始化 V1 物理引擎。
        
        :param k_coupling_const: 振荡器耦合常数 (omega = k * A)。
        :param energy_replenish_rate: 每个 Tick 补充的能量。
        :param base_learning_rate: 贝叶斯更新的基础步长。
        :param chemical_decay_rate: 化学波的自然衰减率 (0.0-1.0)。
        """
        print("PhysicsKernelV1: Initializing (Precision-Weighted)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 物理常数 ---
        self.K_COUPLING_CONST = k_coupling_const
        self.ENERGY_REPLENISH_RATE = energy_replenish_rate
        self.BASE_LEARNING_RATE = base_learning_rate
        self.CHEMICAL_DECAY_RATE = chemical_decay_rate

        # 在类常量区加两行（已可外部注入）
        self.BETA_HOLOGRAPHIC: float = 0.8   # 全息面积系数
        self.SIGMA_VACUUM: float = 0.05      # 真空零点幅
        
        self.EPSILON = 1e-9
        
        print(f"PhysicsKernelV1: Ready. Device: {self.device}")


    @torch.no_grad()
    def execute(
     self,
     lsb_tensor: torch.Tensor,
     sensor_input_tensor: torch.Tensor,
     dt: float,
     schedule_table: Optional[torch.Tensor] = None,   # [num_lncs] 0/1
     dt_scale_table: Optional[torch.Tensor] = None    # [num_lncs] float
     ) -> None:
        

        """
        执行一个“物理 Tick”。 (在 GPU 上并行运行)
        升级：引入李群时间阻尼、Fisher 信息度规、静态能量代价。
        """
        
        # --- 0. 获取“门控” ---
        active_mask = lsb_tensor[..., schema.IS_ACTIVE_INDEX] == 1.0

        if schedule_table is not None and dt_scale_table is not None:
           # 获取每个格点所属的 LNC id（float->long）
           lnc_ids = lsb_tensor[..., schema.LNC_ID_INDEX].long()
           # 映射到每格的是否更新（0/1）与 dt 缩放
           should_update = schedule_table[lnc_ids]
           local_dt_scale = dt_scale_table[lnc_ids]
           scheduled_mask = active_mask & (should_update == 1.0)
           # 每格有效 dt
           dt_eff = dt * torch.clamp(local_dt_scale, min=0.0)
       
        else:   
            scheduled_mask = active_mask
            dt_eff = torch.full_like(lsb_tensor[..., schema.AMPLITUDE_INDEX], dt)
         



        if not torch.any(active_mask):
            return
            
        # --- 1. 更新内在时间 (Time Dynamics) ---
        duration_t = lsb_tensor[..., schema.ACTIVE_DURATION_INDEX]
        
        # 对活跃神经元：累加 dt；非活跃：指数衰减恢复
        duration_new = duration_t.clone()
        duration_new[active_mask] += dt
        duration_new[~active_mask] *= 0.9  # 非活跃区域缓慢恢复
        
        # 计算李群时间阻尼因子 (0~1)
        damping = self._calculate_lie_group_damping(duration_new[active_mask])
        
        # --- 2. 读取 T-1 状态 ---
        a_t_minus_1 = lsb_tensor[..., schema.AMPLITUDE_PREV_INDEX]
        phase_t_minus_1 = lsb_tensor[..., schema.PHASE_PREV_INDEX]
        
        k_t = lsb_tensor[..., schema.BELIEF_SHAPE_K_INDEX]
        theta_rate_t = lsb_tensor[..., schema.BELIEF_RATE_THETA_INDEX]
        
        confidence_t_minus_1 = lsb_tensor[..., schema.DERIVED_CONFIDENCE_PREV_INDEX]
        top_down_prediction = lsb_tensor[..., schema.TOP_DOWN_PREDICTION_INDEX]
        predicted_nt_amp = lsb_tensor[..., schema.PREDICTED_NT_AMPLITUDE_INDEX]
        nt_amplitude = lsb_tensor[..., schema.NT_AMPLITUDE_INDEX]


        #读取提案缓冲（活动掩码下）
        proposal_topdown = lsb_tensor[..., schema.PROPOSAL_TOP_DOWN_INDEX]
        proposal_precision = lsb_tensor[..., schema.PROPOSAL_PRECISION_DELTA_INDEX]
        proposal_phase = lsb_tensor[..., schema.PROPOSAL_PHASE_PROPOSAL_INDEX]
        proposal_decay = lsb_tensor[..., schema.PROPOSAL_DECAY_INDEX]

        #针对活动区域构造有效值
        p_topdown_eff = proposal_topdown[active_mask]
        p_precision_eff = proposal_precision[active_mask]
        p_phase_eff = proposal_phase[active_mask]

        #化学精度门控（高精度=高权重；也可用 sigmoid/softmax）
        precision_gate = torch.clamp(nt_amplitude[active_mask], 0.0, 1.0)

        # 融合提案（加权平均）
        fused_topdown = precision_gate * p_topdown_eff
        fused_precision_delta = precision_gate * p_precision_eff
        fused_phase_delta = precision_gate * p_phase_eff
        
        # --- 3. [定律 2: 贝叶斯学习 & 预测编码] (重构) ---
        k_new, theta_rate_new, alpha_new, \
        confidence_new, ae_ff, \
        sensory_pe, chemical_pe, g_total, trace_write = \
            self._execute_law_2_bayesian_learning(
                active_mask, 
                k_t, theta_rate_t, 
        k_new, theta_rate_new, alpha_new, \
        confidence_new, ae_ff, \
        sensory_pe, chemical_pe = \
            self._execute_law_2_bayesian_learning(
                active_mask, 
                k_t, theta_rate_t, 
                confidence_t_minus_1, 
                top_down_prediction,
                nt_amplitude, predicted_nt_amp, 
                lsb_tensor, sensor_input_tensor,
                fused_topdown, fused_precision_delta
            )

        # 持续学习最小闭环：显式 residual + trace。
        residual_energy = torch.abs(sensory_pe)
        trace_strength = torch.clamp(torch.maximum(residual_energy, trace_write), 0.0, 1.0)
        trace_strength = torch.clamp(residual_energy, 0.0, 1.0)
        high_residual_mask = residual_energy > 0.5
        k_new[high_residual_mask] = torch.lerp(
            k_t[active_mask][high_residual_mask],
            k_new[high_residual_mask],
            0.3,
        )
        
        # --- 4. [定律 1: 振荡器物理] (升级) ---
        # 振幅演化后乘以 damping，实现注意力随时间衰减
        a_new, phase_new, omega_new = self._execute_law_1_oscillator(
            active_mask, a_t_minus_1, phase_t_minus_1, alpha_new, dt, damping
        )

        # --- 5. [定律 3: 时觉感受器] ---
        amplitude_delta, phase_delta = self._execute_law_3_temporal_sense(
            active_mask, a_new, a_t_minus_1, phase_new, phase_t_minus_1
        )
        
        # --- 6. [定律 4: 化学动力学] ---
        nt_amp_new, nt_phase_new = self._execute_law_4_chemical_dynamics(
            active_mask, 
            lsb_tensor[..., schema.NT_AMPLITUDE_INDEX],
            lsb_tensor[..., schema.NT_PHASE_INDEX],
            dt
        )

        # --- 7. [定律 5: 能量物理] (重构) ---
        # 静态代谢 + 动力学阻力 + 动作消耗
        static_metabolic_rate = 0.05
        cost_static = k_new * static_metabolic_rate
        cost_resistance = (1.0 - damping) * a_new * 0.1
        cost_kinetic = (torch.abs(amplitude_delta) + torch.abs(phase_delta)) * 0.1
        total_energy_cost = cost_static + cost_resistance + cost_kinetic
        
        energy_t = lsb_tensor[..., schema.ENERGY_COST_INDEX]
        energy_new = energy_t[active_mask] - total_energy_cost + (self.ENERGY_REPLENISH_RATE * dt)
        energy_new.clamp_min_(0.0)  # 能量下限保护
        
        # 能量耗尽时强制失活（可选）
        collapse_mask = energy_new <= 0.01
        lsb_tensor[..., schema.IS_ACTIVE_INDEX][active_mask][collapse_mask] = 0.0
        
        # --- 8. [写入 T] (就地修改 LSB) ---
        lsb_tensor[..., schema.AMPLITUDE_INDEX][active_mask] = a_new
        lsb_tensor[..., schema.PHASE_INDEX][active_mask] = phase_new
        lsb_tensor[..., schema.BELIEF_SHAPE_K_INDEX][active_mask] = k_new
        lsb_tensor[..., schema.BELIEF_RATE_THETA_INDEX][active_mask] = theta_rate_new
        lsb_tensor[..., schema.AMPLITUDE_PREV_INDEX][active_mask] = a_new
        lsb_tensor[..., schema.PHASE_PREV_INDEX][active_mask] = phase_new
        lsb_tensor[..., schema.DERIVED_ALPHA_INDEX][active_mask] = alpha_new
        lsb_tensor[..., schema.DERIVED_OMEGA_INDEX][active_mask] = omega_new
        lsb_tensor[..., schema.DERIVED_CONFIDENCE_INDEX][active_mask] = confidence_new
        lsb_tensor[..., schema.DERIVED_CONFIDENCE_PREV_INDEX][active_mask] = confidence_new
        lsb_tensor[..., schema.DERIVED_AE_FF_INDEX][active_mask] = ae_ff
        lsb_tensor[..., schema.DERIVED_AMPLITUDE_DELTA_INDEX][active_mask] = amplitude_delta
        lsb_tensor[..., schema.DERIVED_PHASE_DELTA_INDEX][active_mask] = phase_delta
        lsb_tensor[..., schema.PREDICTION_ERROR_INDEX][active_mask] = sensory_pe
        lsb_tensor[..., schema.NT_PREDICTION_ERROR_INDEX][active_mask] = chemical_pe
        lsb_tensor[..., schema.RESIDUAL_ENERGY_INDEX][active_mask] = residual_energy
        lsb_tensor[..., schema.TRACE_STRENGTH_INDEX][active_mask] = trace_strength
        lsb_tensor[..., schema.TRACE_DECAY_INDEX][active_mask] = torch.clamp(0.90 + 0.09 * (1.0 - g_total), 0.90, 0.99)
        lsb_tensor[..., schema.TRACE_DECAY_INDEX][active_mask] = 0.95
        lsb_tensor[..., schema.NT_AMPLITUDE_INDEX][active_mask] = nt_amp_new
        lsb_tensor[..., schema.NT_PHASE_INDEX][active_mask] = nt_phase_new
        lsb_tensor[..., schema.ENERGY_COST_INDEX][active_mask] = energy_new
        lsb_tensor[..., schema.ACTIVE_DURATION_INDEX][active_mask] = duration_new[active_mask]
        lsb_tensor[..., schema.LAST_UPDATE_TICK_INDEX][active_mask] = lsb_tensor[..., schema.LAST_UPDATE_TICK_INDEX][active_mask] + 1
        
        current_tick = lsb_tensor[..., schema.LAST_UPDATE_TICK_INDEX][active_mask]
        lsb_tensor[..., schema.LAST_UPDATE_TICK_INDEX][active_mask] = current_tick + 1
        
        decay_factor = torch.clamp(proposal_decay[active_mask], 0.0, 1.0)

        # [修改开始]：增加数值稳定性保护
        # 防止提案通道因高频写入而溢出
        lsb_tensor[..., schema.PROPOSAL_TOP_DOWN_INDEX][active_mask].clamp_(-5.0, 5.0)
        lsb_tensor[..., schema.PROPOSAL_PRECISION_DELTA_INDEX][active_mask].clamp_(-1.0, 1.0)
        
        # 确保能量不为负 (再次检查)
        lsb_tensor[..., schema.ENERGY_COST_INDEX][active_mask].clamp_min_(0.0)
        
        # 确保振幅不爆炸 (物理限制)
        lsb_tensor[..., schema.AMPLITUDE_INDEX][active_mask].clamp_(0.0, 10.0)
        # [修改结束]


        # 对三个提案通道按衰减系数进行寿命衰减
        lsb_tensor[..., schema.PROPOSAL_TOP_DOWN_INDEX][active_mask] *= decay_factor
        lsb_tensor[..., schema.PROPOSAL_PRECISION_DELTA_INDEX][active_mask] *= decay_factor
        lsb_tensor[..., schema.PROPOSAL_PHASE_PROPOSAL_INDEX][active_mask] *= decay_factor

        # 清理极小的残留（避免长期积累微弱噪声）
        small_mask = lsb_tensor[..., schema.PROPOSAL_TOP_DOWN_INDEX][active_mask].abs() < 1e-3
        lsb_tensor[..., schema.PROPOSAL_TOP_DOWN_INDEX][active_mask][small_mask] = 0.0

        # 可选：同时对另外两个通道做清理（建议一并加上）
        small_mask_prec = lsb_tensor[..., schema.PROPOSAL_PRECISION_DELTA_INDEX][active_mask].abs() < 1e-3
        lsb_tensor[..., schema.PROPOSAL_PRECISION_DELTA_INDEX][active_mask][small_mask_prec] = 0.0

        small_mask_phase = lsb_tensor[..., schema.PROPOSAL_PHASE_PROPOSAL_INDEX][active_mask].abs() < 1e-3
        lsb_tensor[..., schema.PROPOSAL_PHASE_PROPOSAL_INDEX][active_mask][small_mask_phase] = 0.0

    # --- 新增辅助方法：李群时间阻尼 ---
    def _calculate_lie_group_damping(self, active_duration_t, characteristic_time=0.02):
        """
        计算基于 sqrt(t) 的阻尼因子。
        理论公式: damping ~ |cos(2 * pi * characteristic_time * sqrt(t))|
        """
        lie_time = torch.sqrt(active_duration_t + 1e-9)
        phase_arg = 2 * torch.pi * characteristic_time * lie_time
        damping_factor = torch.abs(torch.cos(phase_arg))
        return damping_factor


    # --- 重构辅助方法：振荡器物理（引入阻尼） ---
    def _execute_law_1_oscillator(self, active_mask, a_prev, phase_prev, alpha, dt_eff, damping_t):
       """
       Law 1: 黑洞-全息振荡器
       dA/dt_eff = α·A - (β·A² + σ)  
       饱和值 A∞ = [α + √(α²-4βσ)]/(2β) ≤ α/β
       """
       A_old = a_prev[active_mask]
    
       # 1. 引力塌缩（信念驱动）
       gravity   = alpha * A_old
    
       # 2. 全息饱和 / 霍金辐射（即时阻尼）
       holographic = self.BETA_HOLOGRAPHIC * A_old.square() + self.SIGMA_VACUUM
    
       # 3. 净变化（Euler 一步积分即可，βA² 保证负反馈够强）
       dA = gravity - holographic
    
       # 4. 显式更新
       A_raw = A_old + dA * dt_eff
    
       # 5. 再叠加快刀外的慢刀 √t 阻尼（双重保险，可开关）
       A_new = A_raw * damping_t   # damping_t 仍是 |cos(2πf√t)|
       
       # 5. 黑洞封顶后再截断负值（ReLU=霍金辐射底线）
       A_new = torch.relu(A_raw * damping_t)          # ≤0 一律按 0 处理
       omega_new = self.K_COUPLING_CONST * A_new      # 保证 ω≥0
    
       # 6. 相位保持振幅-频率耦合
       omega_new = self.K_COUPLING_CONST * A_new
       phase_new = phase_prev[active_mask] + omega_new * dt_eff
       phase_new = torch.fmod(phase_new, 2 * torch.pi)

       

       return A_new, phase_new, omega_new


    # --- 重构核心方法：贝叶斯学习（引入 Fisher 信息） ---
    def _execute_law_2_bayesian_learning(self, active_mask, k_t, theta_rate_t, 
                                         confidence_prev, top_down_prediction,
                                         nt_amplitude, predicted_nt_amp,
                                         lsb_tensor, sensor_input,fused_topdown, fused_precision_delta):
        """[定律 2: 贝叶斯学习]（重构版）"""
        # 1. 计算激活概率 p (Amplitude^2)
        p = torch.clamp(lsb_tensor[..., schema.AMPLITUDE_INDEX][active_mask].pow(2), 0.01, 0.99)
        
        # 2. 计算 Fisher 信息（黎曼度规）
        fisher_info = 1.0 / (p * (1.0 - p) + self.EPSILON)
        
        # 3. 获取证据与预测误差
        sensor_amp = sensor_input[..., 0] 
        evidence = lsb_tensor[..., schema.SENSORY_INPUT_INDEX][active_mask]
        prior_alpha = k_t[active_mask] / (theta_rate_t[active_mask] + self.EPSILON)
        total_prediction = top_down_prediction[active_mask] + prior_alpha + fused_topdown
        sensory_pe = evidence - total_prediction

        # 悬置门控（最小实现）
        # g_e: 证据门；g_t: 时基门；g_r: 残差门（高残差时降低直接参数更新）
        g_e = torch.clamp(torch.abs(evidence), 0.0, 1.0)
        g_t = torch.clamp(1.0 / (1.0 + lsb_tensor[..., schema.ACTIVE_DURATION_INDEX][active_mask]), 0.0, 1.0)
        residual_prev = torch.clamp(torch.abs(lsb_tensor[..., schema.PREDICTION_ERROR_INDEX][active_mask]), 0.0, 1.0)
        g_r = 1.0 - residual_prev
        g_total = torch.clamp(g_e * g_t * g_r, 0.0, 1.0)
        
        # 4. 自然梯度更新（抵抗高信念变化）
        precision_gain = torch.relu(nt_amplitude[active_mask]+ fused_precision_delta)
        belief_update = ((precision_gain * sensory_pe) / (fisher_info + self.EPSILON)) * g_total
        k_new = k_t[active_mask] + (belief_update * self.BASE_LEARNING_RATE)
        k_new.clamp_min_(0.1)
        
        theta_rate_new = theta_rate_t[active_mask]
        alpha_new = k_new / (theta_rate_new + self.EPSILON)
        confidence_new = k_new / (theta_rate_new.pow(2) + self.EPSILON)
        ae_ff = confidence_new - confidence_prev[active_mask]
        
        chemical_pe = nt_amplitude[active_mask] - predicted_nt_amp[active_mask]
        trace_write = torch.abs(sensory_pe) * (1.0 - g_total)


        
        return k_new, theta_rate_new, alpha_new, confidence_new, ae_ff, sensory_pe, chemical_pe, g_total, trace_write


    # --- 其余辅助方法（时觉、化学）保持原有逻辑 ---
    def _execute_law_3_temporal_sense(self, active_mask, a_new, a_prev, phase_new, phase_prev):
        """[定律 3: 时觉感受器]（未变）"""
        amplitude_delta = a_new - a_prev[active_mask]
        phase_delta = phase_new - phase_prev[active_mask]
        return amplitude_delta, phase_delta

    def _execute_law_4_chemical_dynamics(self, active_mask, nt_amp, nt_phase, dt):

        """[定律 4: 化学动力学]"""
        # V1: 简单的指数衰减 (模拟酶解)
        # 如果没有新的注入，化学波会随时间消散
        
        # Amp_new = Amp_old * Decay_Rate
        # 注意：衰减率需要根据 dt 调整
        decay_factor = math.pow(self.CHEMICAL_DECAY_RATE, dt * 100) # 假设 DecayRate 是每 10ms 的
        nt_amp_new = nt_amp[active_mask] * decay_factor
        
        # 相位通常保持不变，直到被新波覆盖或完全消散
        nt_phase_new = nt_phase[active_mask]
        
        # 如果振幅太小，重置相位为0 (清理噪声)
        reset_mask = nt_amp_new < 0.01
        nt_phase_new[reset_mask] = 0.0
        nt_amp_new[reset_mask] = 0.0
        
        return nt_amp_new, nt_phase_new
