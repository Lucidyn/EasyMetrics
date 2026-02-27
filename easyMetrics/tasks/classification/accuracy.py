"""
准确率 (Accuracy) 指标实现。
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
from easyMetrics.core.base import Metric


class Accuracy(Metric):
    """
    计算分类任务的准确率 (Accuracy)。
    支持二分类、多分类场景。多标签场景下按样本级精确匹配计算。
    """

    def __init__(self, num_classes: Optional[int] = None):
        """
        参数:
            num_classes: 类别数量。如果为 None，则从数据中自动推断。
        """
        super().__init__()
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0

    def reset(self):
        """重置指标状态。"""
        self.correct = 0
        self.total = 0

    def _ensure_numpy_array(self, data, dtype=None):
        """确保输入为 numpy 数组。"""
        if isinstance(data, np.ndarray):
            return data.astype(dtype) if dtype is not None else data
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=dtype)
        elif np.isscalar(data):
            return np.array([data], dtype=dtype)
        return np.array(data, dtype=dtype)

    def update(self, preds: Union[List[Any], np.ndarray], target: Union[List[Any], np.ndarray]):
        """
        更新指标状态。

        参数:
            preds: 预测结果，可以是类别索引或概率数组。
            target: 真实标签。
        """
        preds = self._ensure_numpy_array(preds)
        target = self._ensure_numpy_array(target)

        if target.ndim == 2:
            # 多标签：样本级精确匹配
            if preds.dtype == float or preds.ndim == 2:
                preds_binary = (preds >= 0.5).astype(int)
            else:
                preds_binary = preds
            correct = np.all(preds_binary == target, axis=1)
        else:
            # 单标签
            if preds.ndim == 2:
                preds = np.argmax(preds, axis=1)
            elif preds.ndim == 1 and preds.dtype == float:
                preds = (preds >= 0.5).astype(int)
            correct = (preds == target)

        self.correct += int(np.sum(correct))
        self.total += len(target)

    def compute(self) -> Dict[str, float]:
        """计算准确率。"""
        if self.total == 0:
            return {"accuracy": 0.0}
        return {"accuracy": self.correct / self.total}
