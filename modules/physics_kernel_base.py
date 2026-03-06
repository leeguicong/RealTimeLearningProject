#
# 文件: modules/physics_kernel_base.py
# 职责: 定义“物理引擎” (PhysicsKernel) 的抽象基类 (接口)。
#
# 这是一个“合约”，它规定了所有“物理定律”实现
# (例如 PhysicsKernel_V1) 必须遵守的共同接口。
#
# 这使得“主循环”(模块7) 可以与“物理定律”解耦。
#

import abc
import torch

class IPhysicsKernel(abc.ABC):
    """
    物理引擎抽象基类 (IPhysicsKernel Interface)。
    
    “主循环”(模块7) 将在每个 Tick 调用此接口的 `execute` 方法。
    """

    @abc.abstractmethod
    def execute(
        self, 
        lsb_tensor: torch.Tensor, 
        sensor_input_tensor: torch.Tensor, 
        dt: float
    ) -> None:
        """
        执行一个“物理 Tick”。
        
        这个方法将“就地”(in-place) 修改 lsb_tensor，以应用
        “振荡器”、“贝叶斯”和“时觉”等定律。
        
        :param lsb_tensor: torch.Tensor
            大静态缓冲区 (LSB) 的主张量。
            形状为 (W, H, N)，例如 (2048, 2048, 2048)。
            此张量将被*就地修改*。
            
        :param sensor_input_tensor: torch.Tensor
            一个较小的张量，包含了“模块1”准备的“感官输入”。
            这是“贝叶斯推断”所需的“证据”(evidence)。
            它的形状将由“模块1”和 Kernel 协商确定。
            
        :param dt: float
            时间增量 (Delta Time)，即此 Tick 所代表的秒数。
            (例如 0.001)
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """
        (可选) 返回此物理引擎的名称，用于日志记录。
        """
        return self.__class__.__name__