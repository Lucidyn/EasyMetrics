"""
EasyMetrics: A simple and extensible metrics evaluation platform.
"""
__version__ = "0.4.2"

# 导出检测任务指标
from .tasks.detection.map import MeanAveragePrecision

# 导出分类任务指标
from .tasks.classification import F1Score, AUC

# 导出统一评估接口
from .tasks import evaluate

__all__ = ['MeanAveragePrecision', 'F1Score', 'AUC', 'evaluate']
