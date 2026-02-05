from typing import Any, Dict, List, Optional, Tuple
from .map import MeanAveragePrecision
from .format_converter import DetectionFormatConverter

def evaluate_detection(
    preds: List[Any], 
    targets: List[Any], 
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
    使用 COCO 风格的指标评估目标检测结果。

    参数:
        preds (List[Any]): 每张图片的预测结果列表。
        targets (List[Any]): 每张图片的真实标签列表。
        metrics (Optional[List[str]]): 需要返回的特定指标列表。
        n_jobs (int): 并行计算线程数。默认为 1。-1 表示使用所有核心。
        score_criteria (Optional[List[Tuple[float, float]]]): 
            计算指定 IoU 和 精度下的最佳置信度阈值。
            格式: [(iou_thresh, min_precision), ...]
            例如: [(0.5, 0.9)]
        format (str): 输入数据的格式，当 pred_format 和 target_format 未指定时使用。
                    支持 "coco", "voc", "yolo", "custom"。
        pred_format (Optional[str]): 预测结果的格式，优先级高于 format。
        target_format (Optional[str]): 真实标签的格式，优先级高于 format。
        progress (bool): 是否显示进度条。默认为 True。
        **kwargs: 额外的转换参数。
            - image_size: YOLO 格式需要的图像尺寸 (width, height)
            - custom_converter: 自定义格式的转换函数

    返回:
        Dict[str, float]: 包含计算指标的字典。
    """
    # 转换输入格式
    converted_preds, converted_targets = DetectionFormatConverter.convert(
        preds, targets, 
        format=format,
        pred_format=pred_format,
        target_format=target_format,
        **kwargs
    )
    
    # 初始化指标计算器
    metric = MeanAveragePrecision()
    
    # 更新数据
    metric.update(converted_preds, converted_targets)
    
    # 计算所有指标
    all_results = metric.compute(n_jobs=n_jobs, score_criteria=score_criteria, progress=progress)
    
    # 如果请求了特定指标，进行筛选
    if metrics:
        filtered_results = {}
        for k in metrics:
            if k in all_results:
                filtered_results[k] = all_results[k]
        return filtered_results
    
    return all_results
