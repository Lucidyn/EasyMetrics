"""
Pytest 配置和共享 fixtures。
"""
import numpy as np
import pytest


@pytest.fixture
def detection_preds_targets():
    """检测任务测试数据 - 完美预测和部分错误预测。"""
    preds_1 = {
        'boxes': np.array([[10, 10, 50, 50]], dtype=float),
        'scores': np.array([0.95], dtype=float),
        'labels': np.array([0], dtype=int)
    }
    targets_1 = {
        'boxes': np.array([[10, 10, 50, 50]], dtype=float),
        'labels': np.array([0], dtype=int)
    }
    preds_2 = {
        'boxes': np.array([
            [100, 100, 150, 150],
            [0, 0, 20, 20]
        ], dtype=float),
        'scores': np.array([0.9, 0.6], dtype=float),
        'labels': np.array([0, 0], dtype=int)
    }
    targets_2 = {
        'boxes': np.array([
            [100, 100, 150, 150],
            [200, 200, 250, 250]
        ], dtype=float),
        'labels': np.array([0, 0], dtype=int)
    }
    return [preds_1, preds_2], [targets_1, targets_2]


@pytest.fixture
def classification_binary_data():
    """二分类测试数据。"""
    preds = [0.8, 0.3, 0.6, 0.9, 0.2]
    targets = [1, 0, 1, 1, 0]
    return preds, targets


@pytest.fixture
def classification_multiclass_data():
    """多分类测试数据。"""
    preds = [
        [0.9, 0.05, 0.05],  # 类别 0
        [0.1, 0.8, 0.1],    # 类别 1
        [0.2, 0.3, 0.5],    # 类别 2
        [0.8, 0.1, 0.1]     # 类别 0
    ]
    targets = [0, 1, 2, 0]
    return preds, targets


@pytest.fixture
def classification_multilabel_data():
    """多标签分类测试数据。"""
    preds = [
        [0.9, 0.8, 0.1],
        [0.2, 0.6, 0.7],
        [0.8, 0.3, 0.2],
        [0.1, 0.2, 0.3]
    ]
    targets = [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 0, 0]
    ]
    return preds, targets
