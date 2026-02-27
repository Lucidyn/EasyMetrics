"""
EasyMetrics: A simple and extensible metrics evaluation platform.
"""
__version__ = "0.4.4"

# 导出检测任务指标
from .tasks.detection.map import MeanAveragePrecision

# 导出分类任务指标
from .tasks.classification import F1Score, AUC, Accuracy

# 导出核心评估接口
from .tasks import evaluate, evaluate_classification
from .tasks.detection.interface import evaluate_detection

__all__ = [
    'MeanAveragePrecision', 'F1Score', 'AUC', 'Accuracy',
    'evaluate', 'evaluate_detection', 'evaluate_classification'
]
