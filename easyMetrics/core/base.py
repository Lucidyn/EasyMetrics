from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Metric(ABC):
    """
    所有指标的抽象基类。
    """
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        """
        重置指标的内部状态。
        """
        pass

    @abstractmethod
    def update(self, preds: Any, target: Any):
        """
        使用新的预测值和真实值更新指标状态。
        
        参数:
            preds: 模型的预测结果。
            target: 真实标签 (Ground Truth)。
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """
        计算最终的指标值。
        
        返回:
            计算出的指标值。
        """
        pass
