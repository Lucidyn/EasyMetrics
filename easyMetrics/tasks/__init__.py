"""
任务评估模块。
"""

from typing import Any, Dict, List, Optional, Tuple, Union

# 导入检测任务评估
from .detection.interface import evaluate_detection

# 导入分类任务评估
from .classification.f1_score import F1Score
from .classification.auc import AUC

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
        # 使用检测任务评估
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
        # 使用分类任务评估
        return _evaluate_classification(
            preds=preds,
            targets=targets,
            metrics=metrics,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的任务类型: {task}")

def _auto_detect_task(preds: Any, targets: Any) -> str:
    """
    根据输入数据自动判断任务类型。
    """
    # 检查是否为检测任务格式
    if isinstance(preds, list):
        if len(preds) > 0:
            first_pred = preds[0]
            if isinstance(first_pred, dict):
                # 检测任务格式通常包含 'boxes', 'scores', 'labels'
                if all(key in first_pred for key in ['boxes', 'scores', 'labels']):
                    return "detection"
                # 或者包含 'boxes', 'labels'（真值）
                elif 'boxes' in first_pred and 'labels' in first_pred:
                    return "detection"
    
    # 默认视为分类任务
    return "classification"

def _evaluate_classification(
    preds: Union[List[Any], Any], 
    targets: Union[List[Any], Any], 
    metrics: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    分类任务评估。
    """
    from .classification import F1Score, AUC
    
    results = {}
    
    # 计算F1 Score
    if metrics is None or any(m in metrics for m in ['f1', 'precision', 'recall']):
        f1_metric = F1Score(**kwargs)
        f1_metric.update(preds, targets)
        f1_results = f1_metric.compute()
        results.update(f1_results)
    
    # 计算AUC
    if metrics is None or 'auc' in metrics:
        auc_metric = AUC(**kwargs)
        auc_metric.update(preds, targets)
        auc_results = auc_metric.compute()
        results.update(auc_results)
    
    # 筛选指定指标
    if metrics:
        filtered_results = {}
        for k in metrics:
            if k in results:
                filtered_results[k] = results[k]
        return filtered_results
    
    return results
