# 职责: 定义神经元特征向量索引。
# 设计目标:
# 1) 兼容现有 PhysicsKernelV1/LNCManager 使用的字段；
# 2) 对齐持续学习文档中的关键概念（Residual/Trace/Prior）。
from typing import Final

# ==================================================
# 1. 基础状态（现有运行路径依赖）
# ==================================================
FEATURE_SPECTRUM_INDEX: Final[int] = 0
SPATIAL_TRANSFORM_INDEX: Final[int] = 1
BELIEF_SHAPE_K_INDEX: Final[int] = 2
BELIEF_RATE_THETA_INDEX: Final[int] = 3
ENERGY_COST_INDEX: Final[int] = 4
LNC_ID_INDEX: Final[int] = 5
IS_ACTIVE_INDEX: Final[int] = 6
SENSORY_INPUT_INDEX: Final[int] = 7

AMPLITUDE_INDEX: Final[int] = 8
PHASE_INDEX: Final[int] = 9
AMPLITUDE_PREV_INDEX: Final[int] = 10
PHASE_PREV_INDEX: Final[int] = 11

TOP_DOWN_PREDICTION_INDEX: Final[int] = 12
PREDICTED_NT_AMPLITUDE_INDEX: Final[int] = 13
NT_AMPLITUDE_INDEX: Final[int] = 14
NT_PHASE_INDEX: Final[int] = 15

ACTIVE_DURATION_INDEX: Final[int] = 16
LAST_UPDATE_TICK_INDEX: Final[int] = 17

# ==================================================
# 2. 派生观测量（Law1~Law4）
# ==================================================
DERIVED_ALPHA_INDEX: Final[int] = 18
DERIVED_OMEGA_INDEX: Final[int] = 19
DERIVED_CONFIDENCE_INDEX: Final[int] = 20
DERIVED_CONFIDENCE_PREV_INDEX: Final[int] = 21
DERIVED_AE_FF_INDEX: Final[int] = 22
DERIVED_AMPLITUDE_DELTA_INDEX: Final[int] = 23
DERIVED_PHASE_DELTA_INDEX: Final[int] = 24
PREDICTION_ERROR_INDEX: Final[int] = 25
NT_PREDICTION_ERROR_INDEX: Final[int] = 26

# ==================================================
# 3. 提案缓冲（Suspension-friendly）
# ==================================================
PROPOSAL_TOP_DOWN_INDEX: Final[int] = 27
PROPOSAL_PRECISION_DELTA_INDEX: Final[int] = 28
PROPOSAL_PHASE_PROPOSAL_INDEX: Final[int] = 29
PROPOSAL_DECAY_INDEX: Final[int] = 30

# ==================================================
# 4. 持续学习扩展（Phase 1 最小槽位）
# ==================================================
# Residual（不可解释部分）
RESIDUAL_ENERGY_INDEX: Final[int] = 31
RESIDUAL_SIG_0_INDEX: Final[int] = 32
RESIDUAL_SIG_1_INDEX: Final[int] = 33
RESIDUAL_SIG_2_INDEX: Final[int] = 34

# Eligibility traces（痕迹）
TRACE_STRENGTH_INDEX: Final[int] = 35
TRACE_DECAY_INDEX: Final[int] = 36

# Prior channel（结构先验）
PRIOR_BIAS_INDEX: Final[int] = 37
PRIOR_STABILITY_INDEX: Final[int] = 38

NEURON_ATTR_DIM: Final[int] = 39
