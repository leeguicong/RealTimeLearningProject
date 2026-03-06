# 文件: modules/lnc_schema.py
# 职责: 定义「局部神经元集群容器」属性索引。
# 改造说明:
# - 保留旧版调度字段 (stride/dt_scale/group/next_tick)
# - 新增持续学习关键槽位: Key / Residual / Frame / Ghost links
from typing import Final

# ==================================================
# 1. 几何容器
# ==================================================
LNC_H_START_INDEX: Final[int] = 0
LNC_W_START_INDEX: Final[int] = 1
LNC_H_SIZE_INDEX: Final[int] = 2
LNC_W_SIZE_INDEX: Final[int] = 3

# ==================================================
# 2. 激活与生存统计
# ==================================================
LNC_ACTIVATION_SPECTRUM_INDEX: Final[int] = 4
LNC_BIRTH_TICK_INDEX: Final[int] = 5
LNC_LAST_UPDATE_TICK_INDEX: Final[int] = 6
LNC_MEMBER_COUNT_INDEX: Final[int] = 7
LNC_CUM_ENERGY_COST_INDEX: Final[int] = 8

# ==================================================
# 3. 扩展区（从 9 开始）
# ==================================================
LNC_EXT_BASE_INDEX: Final[int] = 9

# 3.1 调度字段（兼容既有代码）
LNC_UPDATE_STRIDE_INDEX: Final[int] = LNC_EXT_BASE_INDEX + 0
LNC_DT_SCALE_INDEX: Final[int] = LNC_EXT_BASE_INDEX + 1
LNC_GROUP_ID_INDEX: Final[int] = LNC_EXT_BASE_INDEX + 2
LNC_NEXT_UPDATE_TICK_INDEX: Final[int] = LNC_EXT_BASE_INDEX + 3

# 3.2 Key（路由）
LNC_KEY_CONFIDENCE_INDEX: Final[int] = LNC_EXT_BASE_INDEX + 4
LNC_KEY_SIG_BASE_INDEX: Final[int] = LNC_EXT_BASE_INDEX + 5
LNC_KEY_SIG_DIM: Final[int] = 8

# 3.3 Residual（悬置）
LNC_RESIDUAL_ENERGY_INDEX: Final[int] = LNC_KEY_SIG_BASE_INDEX + LNC_KEY_SIG_DIM
LNC_RESIDUAL_SIG_BASE_INDEX: Final[int] = LNC_RESIDUAL_ENERGY_INDEX + 1
LNC_RESIDUAL_SIG_DIM: Final[int] = 8
LNC_RESIDUAL_MEAN_INDEX: Final[int] = LNC_RESIDUAL_SIG_BASE_INDEX + LNC_RESIDUAL_SIG_DIM
LNC_RESIDUAL_VAR_INDEX: Final[int] = LNC_RESIDUAL_MEAN_INDEX + 1

# 3.4 Frame（参考系）
LNC_FRAME_ALPHA_INDEX: Final[int] = LNC_RESIDUAL_VAR_INDEX + 1
LNC_FRAME_PHI_INDEX: Final[int] = LNC_FRAME_ALPHA_INDEX + 1
LNC_FRAME_DELTA_INDEX: Final[int] = LNC_FRAME_PHI_INDEX + 1
LNC_FRAME_SIGMA_INDEX: Final[int] = LNC_FRAME_DELTA_INDEX + 1

# 3.5 Ghost links（最近关联 id）
LNC_GHOST_LINK_0_INDEX: Final[int] = LNC_FRAME_SIGMA_INDEX + 1
LNC_GHOST_LINK_1_INDEX: Final[int] = LNC_GHOST_LINK_0_INDEX + 1
LNC_GHOST_LINK_2_INDEX: Final[int] = LNC_GHOST_LINK_1_INDEX + 1

LNC_ATTR_DIM: Final[int] = LNC_GHOST_LINK_2_INDEX + 1
