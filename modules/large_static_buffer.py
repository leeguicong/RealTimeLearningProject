#
# 文件: modules/large_static_buffer.py
# 职责: 管理和分配“大静态缓冲区” (LargeStaticBuffer, LSB)。
#       这是AI核心状态的“GPU内存池”。
#

import torch
from typing import Optional

# 导入我们的“神经元”合约，以确保维度一致
from . import neuron_schema

# --- 1. 定义缓冲区的全局维度 (从 Schema 导入) ---

# 2D 计算网格的维度
GRID_WIDTH: int = 2048
GRID_HEIGHT: int = 2048

# "神经元"特征向量的维度 (来自 schema)
# 这证实了我们的总形状是 (2048, 2048, 2048)
FEATURE_DIM: int = neuron_schema.TOTAL_FEATURE_DIM

# 我们商定的计算和存储精度
BUFFER_DTYPE: torch.dtype = torch.float16


class LargeStaticBuffer:
    """
    大静态缓冲区 (LSB) 管理器。
    
    它封装了在GPU上分配、管理和“切片”(Slice) 
    我们的主状态张量 (Tensor) 的逻辑。
    
    职责：
    - 在GPU上分配一个巨大的 (W, H, N) 张量。
    - 提供一个接口 (`get_slice`)，以“切片”出
      “计算缓冲区”(Computation Buffers, CBs)。
    - 不执行任何计算，它只是一个“内存池”。
    """

    def __init__(self):
        """
        初始化LSB管理器。
        """
        if not torch.cuda.is_available():
            print("="*50)
            print("!! 关键错误: LSB 启动失败 !!")
            print("!! PyTorch 未检测到 CUDA (GPU) !!")
            print("!! 我们的架构必须在NVIDIA GPU上运行。")
            print("="*50)
            raise RuntimeError("CUDA not available. LargeStaticBuffer requires a GPU.")
            
        self._device: torch.device = torch.device("cuda")
        self._buffer: Optional[torch.Tensor] = None
        
        # 从 schema 中读取维度
        self.width: int = GRID_WIDTH
        self.height: int = GRID_HEIGHT
        self.feature_dim: int = FEATURE_DIM
        self.dtype: torch.dtype = BUFFER_DTYPE
        
        print(f"LSB: Manager initialized. Target device: {self._device} (NVIDIA 5090)")
        print(f"LSB: Target grid shape: ({self.width}, {self.height}, {self.feature_dim})")
        print(f"LSB: Target precision: {self.dtype}")

    def allocate(self) -> None:
        """
        在GPU上执行实际的内存分配。
        这是一个昂贵的操作，应在AI启动时调用一次。
        """
        if self._buffer is not None:
            print("LSB Warning: Buffer already allocated.")
            return

        # 计算预期的内存占用 (2 bytes for FP16)
        bytes_per_element = torch.tensor([], dtype=self.dtype).element_size()
        total_elements = self.width * self.height * self.feature_dim
        total_bytes = total_elements * bytes_per_element
        total_gb = total_bytes / (1024 ** 3)
        
        print("="*50)
        print(f"LSB: !! 正在分配内存 !!")
        print(f"LSB: 目标元素: {total_elements:,}")
        print(f"LSB: 预计占用: {total_gb:.2f} GB VRAM... (目标: 24 GB)")
        print("="*50)
        
        try:
            # 关键操作：在GPU上创建并初始化为 0
            # 我们使用 .zeros 来确保内存被“圈占”并清理
            self._buffer = torch.zeros(
                (self.width, self.height, self.feature_dim),
                dtype=self.dtype,
                device=self._device
            )
            print(f"LSB: !! 内存分配成功 !! ({total_gb:.2f} GB 已圈占)")
            
        except torch.cuda.OutOfMemoryError as e:
            print("="*50)
            print("!! 灾难性错误: GPU 内存不足 (OOM) !!")
            print(f"!! 尝试分配 {total_gb:.2f} GB 失败。")
            print(f"!! 您的 24GB VRAM (5090) 是否已被其他程序占用？")
            print(f"!! 错误详情: {e}")
            print("="*50)
            raise RuntimeError(f"Failed to allocate LSB due to OOM: {e}") from e
        except Exception as e:
            print(f"LSB: 发生未知分配错误: {e}")
            raise

    def get_slice(self, x_start: int, y_start: int, width: int, height: int) -> torch.Tensor:
        """
        获取一个“计算缓冲区”(CB) 切片。
        
        这返回一个“视图”(View)，而不是“副本”(Copy)。
        对返回的张量进行的所有修改，都会*直接*写入LSB。
        
        :param x_start: 切片的X轴起始索引
        :param y_start: 切片的Y轴起始索引
        :param width: 切片的宽度
        :param height: 切片的高度
        :return: torch.Tensor, 形状为 (height, width, FEATURE_DIM)
                 注意: PyTorch 的索引顺序是 (H, W, C)
        """
        if self._buffer is None:
            raise RuntimeError("LSB Error: Must call .allocate() before .get_slice().")

        # 边界检查
        x_end = x_start + width
        y_end = y_start + height
        
        if (x_start < 0 or y_start < 0 or
            x_end > self.width or y_end > self.height):
            raise ValueError(
                f"LSB Error: Slice (x={x_start}:{x_end}, y={y_start}:{y_end}) "
                f"is out of bounds for buffer size (W={self.width}, H={self.height})."
            )
            
        # 核心操作：返回一个张量“视图”
        # [..., :] 表示“获取该区域所有的 2048 个特征”
        # 注意: PyTorch 的张量索引通常是 [y, x] 或 [row, col]
        # 所以我们的 (H, W, N) 张量 索引为 [y, x, feature]
        return self._buffer[y_start:y_end, x_start:x_end, :]

    def free(self) -> None:
        """
        在AI关闭时，释放GPU内存。
        """
        if self._buffer is not None:
            print("LSB: Freeing GPU memory...")
            del self._buffer
            self._buffer = None
            torch.cuda.empty_cache() # 清理PyTorch的缓存
            print("LSB: Memory freed.")
        else:
            print("LSB: No memory to free.")