"""
任务评估模块。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# 导入检测任务评估
from .detection.interface import evaluate_detection

# 导入分类任务评估
from .classification import F1Score, AUC, Accuracy

def evaluate(
    preds: Union[List[Any], Any], 
    targets: Union[List[Any], Any], 
    task: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    n_jobs: int = 1,
    score_criteria: Optional[List[Tuple[float, float]]] = None,
    format: str = "coco",
    pred_format: Optional[str] = None,
    target_format: Optional[str] = None,
    progress: bool = True,
    **kwargs
) -> Dict[str, float]:
    """
    统一评估接口，根据任务类型选择合适的评估指标。

    参数:
        preds: 模型预测结果。
        targets: 真实标签。
        task: 任务类型。可选值："detection"、"classification"。
              如果为None，会根据输入数据自动判断。
        metrics: 需要返回的特定指标列表。
        n_jobs: 并行计算线程数。默认为 1。-1 表示使用所有核心。
        score_criteria: 计算指定阈值下的最佳置信度阈值。
        format: 输入数据的格式。
        pred_format: 预测结果的格式，优先级高于 format。
        target_format: 真实标签的格式，优先级高于 format。
        progress: 是否显示进度条。默认为 True。
        **kwargs: 额外的参数。

    返回:
        Dict[str, float]: 包含计算指标的字典。
    """
    # 自动判断任务类型
    if task is None:
        task = _auto_detect_task(preds, targets)
    
    if task == "detection":
        return evaluate_detection(
            preds=preds,
            targets=targets,
            metrics=metrics,
            n_jobs=n_jobs,
            score_criteria=score_criteria,
            format=format,
            pred_format=pred_format,
            target_format=target_format,
            progress=progress,
            **kwargs
        )
    elif task == "classification":
        return evaluate_classification(
            preds=preds,
            targets=targets,
            metrics=metrics,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的任务类型: {task}")

def evaluate_classification(
    preds: Union[List[Any], Any], 
    targets: Union[List[Any], Any], 
    metrics: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    分类任务评估。
    
    参数:
        preds: 模型预测结果。
        targets: 真实标签。
        metrics: 需要返回的特定指标列表。
        **kwargs: 额外的参数。

    返回:
        Dict[str, float]: 包含计算指标的字典。
    """
    results = {}

    # 计算 Accuracy
    if metrics is None or (metrics and 'accuracy' in metrics):
        acc_metric = Accuracy()
        acc_metric.update(preds, targets)
        results.update(acc_metric.compute())

    # 计算F1 Score
    if metrics is None or any(m in metrics for m in ['f1', 'precision', 'recall']):
        f1_kwargs = {k: v for k, v in kwargs.items() if k != 'multi_class'}
        f1_metric = F1Score(**f1_kwargs)
        f1_metric.update(preds, targets)
        results.update(f1_metric.compute())
    
    # 计算AUC
    if metrics is None or 'auc' in metrics:
        auc_metric = AUC(**kwargs)
        auc_metric.update(preds, targets)
        results.update(auc_metric.compute())
    
    # 筛选指定指标
    if metrics:
        return {k: results[k] for k in metrics if k in results}
    
    return results

def _auto_detect_task(preds: Any, targets: Any) -> str:
    """
    根据输入数据自动判断任务类型。
    """
    # 检查是否为检测任务格式
    if isinstance(preds, list) and len(preds) > 0:
        first_pred = preds[0]
        if isinstance(first_pred, dict):
            if 'boxes' in first_pred or any(key in first_pred for key in ['bboxes', 'bounding_boxes', 'detections']):
                return "detection"
        elif isinstance(first_pred, list) and len(first_pred) >= 5:
            return "detection"
    
    # 检查是否为分类任务格式
    if isinstance(preds, (list, np.ndarray)):
        if isinstance(preds, np.ndarray) and preds.ndim == 2:
            return "classification"
        elif (isinstance(preds, list) and len(preds) > 0) or (isinstance(preds, np.ndarray) and preds.ndim == 1):
            first_elem = preds[0] if isinstance(preds, list) else preds[0]
            if isinstance(first_elem, (int, float, np.number)) or (isinstance(first_elem, (list, np.ndarray)) and len(first_elem) > 1):
                return "classification"
    
    # 默认视为分类任务
    return "classification"
