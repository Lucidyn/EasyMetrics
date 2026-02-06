from typing import Any, Dict, List, Optional, Union
import numpy as np
from easyMetrics.core.base import Metric

class F1Score(Metric):
    """
    计算分类任务的F1 Score指标。
    支持二分类、多分类和多标签场景。
    """
    def __init__(self, 
                 average: str = 'macro',
                 num_classes: Optional[int] = None):
        """
        参数:
            average: 平均方式。可选值：
                - 'macro': 宏平均，对每个类别/标签计算F1后取平均值
                - 'micro': 微平均，对所有类别/标签计算总的精确率和召回率后计算F1
                - 'weighted': 加权平均，根据每个类别/标签的样本数加权
            num_classes: 类别数量。如果为None，则从数据中自动推断
        """
        super().__init__()
        self.average = average
        self.num_classes = num_classes
        
        # 重置内部状态
        self.reset()

    def reset(self):
        """
        重置指标的内部状态。
        """
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.class_tp = {}  # 每个类别的真正例
        self.class_fp = {}  # 每个类别的假正例
        self.class_fn = {}  # 每个类别的假负例

    def _ensure_numpy_array(self, data, dtype=None):
        """
        确保输入数据是numpy数组。
        """
        if isinstance(data, np.ndarray):
            return data.astype(dtype) if dtype is not None else data
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=dtype)
        elif np.isscalar(data):
            return np.array([data], dtype=dtype)
        else:
            return np.array(data, dtype=dtype)

    def update(self, preds: Union[List[Any], np.ndarray], target: Union[List[Any], np.ndarray]):
        """
        更新指标状态。
        
        参数:
            preds: 模型预测结果。可以是类别索引、概率数组或二值预测。
            target: 真实标签。可以是类别索引或二值标签矩阵（多标签）。
        """
        # 转换为numpy数组
        preds = self._ensure_numpy_array(preds)
        target = self._ensure_numpy_array(target)
        
        # 处理多标签情况
        if target.ndim == 2:
            # 多标签情况
            self._update_multilabel(preds, target)
        else:
            # 单标签情况
            self._update_singlelabel(preds, target)

    def _update_singlelabel(self, preds: np.ndarray, target: np.ndarray):
        """
        更新单标签分类的指标状态。
        """
        # 处理概率数组（取最大值索引）
        if preds.ndim == 2:
            # 对于概率数组，形状是 (n_samples, n_classes)，而target是 (n_samples,)
            # 所以这里不需要形状一致的检查
            preds = np.argmax(preds, axis=1)
        else:
            # 确保形状一致
            assert preds.shape == target.shape, f"预测和标签形状不一致: {preds.shape} vs {target.shape}"
        
        # 处理二分类情况（如果是概率，转换为0/1）
        if preds.ndim == 1 and preds.dtype == float:
            preds = (preds >= 0.5).astype(int)
        
        # 自动推断类别数量
        if self.num_classes is None:
            all_labels = np.concatenate([preds, target])
            self.num_classes = int(np.max(all_labels)) + 1
        
        # 计算每个类别的TP、FP、FN
        for cls in range(self.num_classes):
            tp = np.sum((preds == cls) & (target == cls))
            fp = np.sum((preds == cls) & (target != cls))
            fn = np.sum((preds != cls) & (target == cls))
            
            self.class_tp[cls] = self.class_tp.get(cls, 0) + tp
            self.class_fp[cls] = self.class_fp.get(cls, 0) + fp
            self.class_fn[cls] = self.class_fn.get(cls, 0) + fn
        
        # 计算总体TP、FP、FN
        self.true_positives = sum(self.class_tp.values())
        self.false_positives = sum(self.class_fp.values())
        self.false_negatives = sum(self.class_fn.values())

    def _update_multilabel(self, preds: np.ndarray, target: np.ndarray):
        """
        更新多标签分类的指标状态。
        """
        # 确保形状一致
        assert preds.shape == target.shape, f"多标签预测和标签形状不一致: {preds.shape} vs {target.shape}"
        
        # 处理概率数组（转换为二值预测）
        if preds.dtype == float:
            preds = (preds >= 0.5).astype(int)
        
        # 自动推断标签数量
        if self.num_classes is None:
            self.num_classes = target.shape[1]
        
        # 计算每个标签的TP、FP、FN
        for label in range(self.num_classes):
            tp = np.sum((preds[:, label] == 1) & (target[:, label] == 1))
            fp = np.sum((preds[:, label] == 1) & (target[:, label] == 0))
            fn = np.sum((preds[:, label] == 0) & (target[:, label] == 1))
            
            self.class_tp[label] = self.class_tp.get(label, 0) + tp
            self.class_fp[label] = self.class_fp.get(label, 0) + fp
            self.class_fn[label] = self.class_fn.get(label, 0) + fn
        
        # 计算总体TP、FP、FN
        self.true_positives = sum(self.class_tp.values())
        self.false_positives = sum(self.class_fp.values())
        self.false_negatives = sum(self.class_fn.values())

    def compute(self) -> Dict[str, float]:
        """
        计算F1 Score指标。
        
        返回:
            包含以下键的字典:
                - f1: F1 Score
                - precision: 精确率
                - recall: 召回率
                - 每个类别/标签的精确率、召回率和F1 Score（如果适用）
        """
        results = {}
        
        # 计算每个类别/标签的指标
        class_precisions = []
        class_recalls = []
        class_f1_scores = []
        
        for cls in range(self.num_classes):
            tp = self.class_tp.get(cls, 0)
            fp = self.class_fp.get(cls, 0)
            fn = self.class_fn.get(cls, 0)
            
            # 计算精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # 计算召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # 计算F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_precisions.append(precision)
            class_recalls.append(recall)
            class_f1_scores.append(f1)
            
            # 添加每个类别/标签的指标
            results[f'precision_{cls}'] = precision
            results[f'recall_{cls}'] = recall
            results[f'f1_{cls}'] = f1
        
        # 计算总体指标
        if self.average == 'micro':
            # 微平均：使用总体TP、FP、FN
            precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0.0
            recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        elif self.average == 'macro':
            # 宏平均：对每个类别/标签的指标取平均
            precision = np.mean(class_precisions)
            recall = np.mean(class_recalls)
            f1 = np.mean(class_f1_scores)
        elif self.average == 'weighted':
            # 加权平均：根据每个类别/标签的样本数加权
            support = [self.class_tp.get(cls, 0) + self.class_fn.get(cls, 0) for cls in range(self.num_classes)]
            total_support = sum(support)
            if total_support > 0:
                weights = [s / total_support for s in support]
                precision = np.average(class_precisions, weights=weights)
                recall = np.average(class_recalls, weights=weights)
                f1 = np.average(class_f1_scores, weights=weights)
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            raise ValueError(f"不支持的平均方式: {self.average}")
        
        # 添加总体指标
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        
        return results
