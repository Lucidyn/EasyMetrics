from typing import Any, Dict, List, Optional, Union
import numpy as np
from easyMetrics.core.base import Metric

class AUC(Metric):
    """
    计算分类任务的AUC（Area Under the ROC Curve）指标。
    支持二分类、多分类和多标签场景。
    """
    def __init__(self, 
                 method: str = 'linear',
                 average: str = 'macro',
                 multi_class: str = 'ovr',
                 num_classes: Optional[int] = None):
        """
        参数:
            method: ROC曲线下面积的计算方法。可选值：
                - 'linear': 线性插值方法
                - 'trapezoidal': 梯形积分方法
            average: 多分类或多标签时的平均方式。可选值：
                - 'macro': 宏平均，对每个类别/标签计算AUC后取平均值
                - 'micro': 微平均，将所有类别/标签的预测和标签合并后计算AUC
                - 'weighted': 加权平均，根据每个类别/标签的样本数加权
            multi_class: 多分类时的处理方法。可选值：
                - 'ovr': One-vs-Rest，为每个类别计算AUC
                - 'ovo': One-vs-One，为每对类别计算AUC
            num_classes: 类别数量。如果为None，则从数据中自动推断
        """
        super().__init__()
        self.method = method
        self.average = average
        self.multi_class = multi_class
        self.num_classes = num_classes
        
        # 重置内部状态
        self.reset()

    def reset(self):
        """
        重置指标的内部状态。
        """
        self.scores = []  # 预测分数或概率
        self.labels = []  # 真实标签

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
            preds: 模型预测结果。可以是概率值、分数或类别索引。
            target: 真实标签。可以是类别索引或二值标签矩阵（多标签）。
        """
        # 转换为numpy数组
        preds = self._ensure_numpy_array(preds, dtype=float)
        target = self._ensure_numpy_array(target, dtype=int)
        
        # 处理概率数组或多标签情况
        if preds.ndim == 2:
            # 如果是多标签，确保形状一致
            if target.ndim == 2:
                assert preds.shape == target.shape, f"多标签预测和标签形状不一致: {preds.shape} vs {target.shape}"
            # 如果是单标签多分类，确保目标是一维数组
            elif target.ndim == 1:
                if self.num_classes is None:
                    self.num_classes = preds.shape[1]
        else:
            # 单标签二分类或多分类
            assert preds.ndim == target.ndim, f"预测和标签维度不一致: {preds.ndim} vs {target.ndim}"
            assert preds.shape == target.shape, f"预测和标签形状不一致: {preds.shape} vs {target.shape}"
        
        # 存储数据
        if preds.ndim == 1:
            self.scores.extend(preds.tolist())
        else:
            self.scores.extend(preds.tolist())
        
        if target.ndim == 1:
            self.labels.extend(target.tolist())
        else:
            self.labels.extend(target.tolist())

    def _compute_roc_curve(self, scores: np.ndarray, labels: np.ndarray):
        """
        计算ROC曲线的TPR和FPR。
        
        返回:
            fpr: 假阳性率数组
            tpr: 真阳性率数组
            thresholds: 阈值数组
        """
        # 排序（从高到低）
        sorted_indices = np.argsort(-scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # 计算累积TP和FP
        cumulative_tp = np.cumsum(sorted_labels)
        cumulative_fp = np.cumsum(1 - sorted_labels)
        
        # 计算总阳性和总阴性
        total_positive = np.sum(labels)
        total_negative = len(labels) - total_positive
        
        # 避免除零错误
        if total_positive == 0 or total_negative == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
        
        # 计算TPR和FPR
        tpr = cumulative_tp / total_positive
        fpr = cumulative_fp / total_negative
        
        # 添加 (0, 0) 点
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        thresholds = np.concatenate([[1.0], sorted_scores])
        
        return fpr, tpr, thresholds

    def _compute_auc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        计算ROC曲线下的面积。
        """
        if self.method == 'linear':
            # 线性插值方法
            auc = 0.0
            for i in range(1, len(fpr)):
                auc += 0.5 * (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1])
        elif self.method == 'trapezoidal':
            # 梯形积分方法
            auc = np.trapz(tpr, fpr)
        else:
            raise ValueError(f"不支持的计算方法: {self.method}")
        
        return auc

    def _compute_binary_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        计算二分类AUC。
        """
        fpr, tpr, _ = self._compute_roc_curve(scores, labels)
        return self._compute_auc(fpr, tpr)

    def _compute_multiclass_auc(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        计算多分类AUC。
        """
        results = {}
        
        # 自动推断类别数量
        if self.num_classes is None:
            self.num_classes = int(np.max(labels)) + 1
        
        if self.multi_class == 'ovr':
            # One-vs-Rest 方法
            class_aucs = []
            for cls in range(self.num_classes):
                # 将当前类别视为正类，其他视为负类
                binary_labels = (labels == cls).astype(int)
                if scores.ndim == 1:
                    # 预测是类别索引，无法计算AUC
                    return {'auc': 0.0}
                else:
                    # 使用当前类别的概率
                    cls_scores = scores[:, cls]
                    auc = self._compute_binary_auc(cls_scores, binary_labels)
                    class_aucs.append(auc)
                    results[f'auc_{cls}'] = auc
        
        elif self.multi_class == 'ovo':
            # One-vs-One 方法
            class_aucs = []
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    # 只考虑属于这两个类别的样本
                    mask = (labels == i) | (labels == j)
                    if not np.any(mask):
                        continue
                    
                    # 过滤样本
                    filtered_scores = scores[mask]
                    filtered_labels = labels[mask]
                    
                    # 将标签转换为二分类
                    binary_labels = (filtered_labels == i).astype(int)
                    
                    # 使用这两个类别的概率
                    cls_scores = filtered_scores[:, i] - filtered_scores[:, j]
                    auc = self._compute_binary_auc(cls_scores, binary_labels)
                    class_aucs.append(auc)
                    results[f'auc_{i}_vs_{j}'] = auc
        
        else:
            raise ValueError(f"不支持的多分类方法: {self.multi_class}")
        
        # 计算平均AUC
        if class_aucs:
            if self.average == 'macro':
                avg_auc = np.mean(class_aucs)
            elif self.average == 'weighted':
                # 计算每个类别的样本数作为权重
                class_counts = np.bincount(labels, minlength=self.num_classes)
                weights = class_counts / len(labels)
                avg_auc = np.average(class_aucs, weights=weights[:len(class_aucs)])
            elif self.average == 'micro':
                # 微平均：将所有类别合并为二分类
                if scores.ndim == 2:
                    # 将多分类转换为二分类（每个类别作为一个样本）
                    all_scores = []
                    all_labels = []
                    for cls in range(self.num_classes):
                        binary_labels = (labels == cls).astype(int)
                        cls_scores = scores[:, cls]
                        all_scores.extend(cls_scores)
                        all_labels.extend(binary_labels)
                    avg_auc = self._compute_binary_auc(np.array(all_scores), np.array(all_labels))
                else:
                    avg_auc = 0.0
            else:
                raise ValueError(f"不支持的平均方式: {self.average}")
        else:
            avg_auc = 0.0
        
        results['auc'] = avg_auc
        return results

    def _compute_multilabel_auc(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        计算多标签AUC。
        """
        results = {}
        num_labels = labels.shape[1]
        
        # 为每个标签计算AUC
        label_aucs = []
        for i in range(num_labels):
            auc = self._compute_binary_auc(scores[:, i], labels[:, i])
            label_aucs.append(auc)
            results[f'auc_label_{i}'] = auc
        
        # 计算平均AUC
        if label_aucs:
            if self.average == 'macro':
                avg_auc = np.mean(label_aucs)
            elif self.average == 'weighted':
                # 计算每个标签的正例数作为权重
                label_counts = np.sum(labels, axis=0)
                weights = label_counts / np.sum(label_counts)
                avg_auc = np.average(label_aucs, weights=weights)
            elif self.average == 'micro':
                # 微平均：将所有标签合并为二分类
                avg_auc = self._compute_binary_auc(scores.flatten(), labels.flatten())
            else:
                raise ValueError(f"不支持的平均方式: {self.average}")
        else:
            avg_auc = 0.0
        
        results['auc'] = avg_auc
        return results

    def compute(self) -> Dict[str, float]:
        """
        计算AUC指标。
        
        返回:
            包含AUC值的字典。
        """
        # 转换为numpy数组
        scores = np.array(self.scores, dtype=float)
        labels = np.array(self.labels, dtype=int)
        
        # 检查数据是否为空
        if len(scores) == 0:
            return {'auc': 0.0}
        
        # 处理不同情况
        if scores.ndim == 1:
            # 二分类或多分类（预测是类别索引）
            if labels.ndim == 1:
                # 检查是否是二分类
                unique_labels = np.unique(labels)
                if set(unique_labels).issubset({0, 1}):
                    # 二分类
                    auc = self._compute_binary_auc(scores, labels)
                    return {'auc': auc}
                else:
                    # 多分类，但预测是类别索引，无法计算AUC
                    return {'auc': 0.0}
            else:
                # 多标签，但预测是一维数组，无法计算AUC
                return {'auc': 0.0}
        else:
            # 多分类或多标签
            if labels.ndim == 1:
                # 多分类
                return self._compute_multiclass_auc(scores, labels)
            else:
                # 多标签
                return self._compute_multilabel_auc(scores, labels)

