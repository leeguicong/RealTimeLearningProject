#
# 文件: modules/output_controller.py
# 职责: 定义“输出-行动模块”(Action Module)的抽象接口和模拟实现。
#
# 这代表了“行动协处理器”(芯片C)的“Python驱动程序”。
# “主脑”(芯片A)上的“行动LNC”将通过此接口发送“行动振荡器”。
#

import abc
import torch
from typing import Tuple, Dict, Any, Optional

class IOutputController(abc.ABC):
    """
    输出控制器抽象基类 (IOutputController Interface)。
    
    这是“主循环”(模块7)与“行动协处理器”(芯片C)
    之间通信的“硬件抽象合约”。
    
    它封装了“振荡器指令”的 P2P DMA 传出，
    以及“物理常量”的反馈。
    """

    @abc.abstractmethod
    def initialize_controller(self, controller_config: Dict[str, Any]) -> bool:
        """
        初始化物理行动硬件 (例如：连接机械臂的 CAN 总线)。
        
        :param controller_config: 一个字典，包含此控制器的特定配置
                                  (例如: {"port": "/dev/ttyUSB0"})
        :return: bool, True 表示初始化成功。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_specification(self) -> Tuple[Tuple[int, ...], torch.dtype]:
        """
        获取此控制器（协处理器）期望接收的“振荡器指令”的数据规格。
        
        “主脑”(芯片A)将据此规格来准备“行动张量”。
        
        :return: (shape, dtype)
            - shape: 一个元组，例如 (7, 2) [Num_Joints, Channels]
                     (代表 7 个关节，2个通道: 振幅A 和 相位theta)
            - dtype: PyTorch 数据类型，例如 torch.float16
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_physical_constants(self) -> torch.Tensor:
        """
        获取此控制器所连接的“物理材料”的“常量偏移”。
        (来自您的“反馈”理论)
        
        这在启动时被调用一次，用于向“主脑”报告
        “身体”的固有物理属性 (例如 弹簧应力, 柔韧性)。
        
        “主脑”上的 LNC 将“观测”这些常量，以学习“身体模型”。
        
        :return: torch.Tensor, 一个包含物理常量的张量。
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def execute_oscillator_command(self, oscillator_action_tensor: torch.Tensor) -> None:
        """
        (异步) 执行一个 Tick 的“行动”指令。
        
        这是此模块的核心驱动方法。
        
        内部逻辑 (被此接口抽象掉)：
        1. (主脑 A -> 协处理器 C) 执行 P2P DMA 传输...
        2. ...将“行动振荡器” (oscillator_action_tensor)
           发送到“行动协处理器”(芯片C)的内存中。
        3. (芯片C) 协处理器接收此“振荡器”张量，
           并将其“反向翻译”为底层的物理动作 (例如 PWM 信号)。
        
        :param oscillator_action_tensor:
            一个位于“主脑”(芯片A) VRAM 上的、
            由“行动LNC”计算出的“振荡器指令”张量。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self) -> None:
        """
        安全关闭行动硬件 (例如：使马达断电)。
        """
        raise NotImplementedError


# --- 模拟实现 (用于测试) ---

class SimulatedMotorController(IOutputController):
    """
    模拟的电机控制器 (例如 机械臂)。
    
    它实现 IOutputController 接口，但*不*需要真实的电机硬件。
    它只是“假装”接收“振荡器”指令，并“报告”
    一个“伪造的”物理常量张量。
    
    这使我们能（在纯模拟中）测试完整的“感知-思考-行动-反馈”循环。
    """
    
    def __init__(self, 
                 sim_spec: Tuple[Tuple[int, ...], torch.dtype],
                 sim_physical_constants: torch.Tensor):
        print("="*50)
        print("!! 警告: 正在加载 *模拟* 输出控制器 (SimulatedMotorController) !!")
        print("!! 未连接到真实的“行动协处理器”。将只在控制台打印动作。")
        print("="*50)
        
        self._spec = sim_spec
        self._physical_constants = sim_physical_constants
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_controller(self, controller_config: Dict[str, Any]) -> bool:
        print(f"SimulatedController: '初始化' 控制器 (配置: {controller_config})... 成功。")
        return True

    def get_specification(self) -> Tuple[Tuple[int, ...], torch.dtype]:
        """
        返回我们在构造时定义的模拟规格。
        """
        return self._spec

    def get_physical_constants(self) -> torch.Tensor:
        """
        返回我们在构造时定义的“伪造”的物理常量张量。
        (实现了您的“常量偏移”反馈)
        """
        return self._physical_constants.to(self._device)

    async def execute_oscillator_command(self, oscillator_action_tensor: torch.Tensor) -> None:
        """
        “模拟”的行动执行。
        
        我们跳过所有 DMA 硬件逻辑，只是“读取”指令并打印到控制台。
        """
        
        # 验证输入张量是否符合规格
        if oscillator_action_tensor.shape != self._spec[0]:
            raise ValueError(f"SimController Error: 传入的行动张量形状 {oscillator_action_tensor.shape} "
                             f"与模拟器规格 {self._spec[0]} 不匹配。")
            
        # “模拟”的行动
        # 我们可以只打印第一个关节的信息作为示例
        # .item() 会从 GPU 复制到 CPU，这在模拟中是OK的
        try:
            joint_0_amplitude = oscillator_action_tensor[0, 0].item()
            joint_0_phase = oscillator_action_tensor[0, 1].item()
            
            print(f"SimController: [TICK] 收到指令... 关节 0: [振幅 A={joint_0_amplitude:.2f}, 相位 \u03B8={joint_0_phase:.2f}]")
        except Exception as e:
            # 捕获可能的张量不在CPU上的错误
            print(f"SimController: [TICK] 收到指令 (张量在 {oscillator_action_tensor.device})")
            
        # (在真实实现中，我们会 `await hardware.dma_transfer()`，
        # 但在这里，操作是瞬时完成的)

    def shutdown(self) -> None:
        print("SimulatedController: '关闭' 控制器 (马达断电)。")
        pass