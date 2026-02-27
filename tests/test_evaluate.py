"""
统一评估接口和集成测试。
"""
import numpy as np
import pytest
from easyMetrics import evaluate, evaluate_detection, evaluate_classification


class TestEvaluate:
    """evaluate 统一接口测试。"""

    def test_auto_detect_classification(self):
        """自动检测分类任务。"""
        preds = [[0.9, 0.1], [0.3, 0.7]]
        targets = [0, 1]
        results = evaluate(preds, targets)
        assert 'f1' in results or 'auc' in results

    def test_auto_detect_detection(self):
        """自动检测检测任务。"""
        preds = [{
            'boxes': [[10, 10, 50, 50]],
            'scores': [0.95],
            'labels': [0]
        }]
        targets = [{
            'boxes': [[10, 10, 50, 50]],
            'labels': [0]
        }]
        results = evaluate(preds, targets)
        assert 'mAP' in results or 'mAP_50' in results

    def test_explicit_classification(self):
        """显式指定分类任务。"""
        preds = [[0.9, 0.1], [0.3, 0.7]]
        targets = [0, 1]
        results = evaluate(preds, targets, task='classification')
        assert 'f1' in results
        assert 'auc' in results

    def test_explicit_detection(self):
        """显式指定检测任务。"""
        preds = [{
            'boxes': [[10, 10, 50, 50]],
            'scores': [0.95],
            'labels': [0]
        }]
        targets = [{
            'boxes': [[10, 10, 50, 50]],
            'labels': [0]
        }]
        results = evaluate(preds, targets, task='detection')
        assert 'mAP' in results

    def test_invalid_task(self):
        """无效任务类型。"""
        with pytest.raises(ValueError, match="不支持的任务类型"):
            evaluate([1, 2], [1, 2], task='invalid')


class TestIntegration:
    """集成测试，验证 examples 中的用例。"""

    def test_examples_flow(self):
        """模拟 examples.py 中的主要流程。"""
        # 分类
        class_preds = [[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]]
        class_targets = [0, 1, 0]
        r1 = evaluate(class_preds, class_targets)
        assert 'f1' in r1 or 'auc' in r1

        # 检测
        det_preds = [{
            'boxes': [[10, 10, 50, 50], [100, 100, 150, 150]],
            'scores': [0.95, 0.9],
            'labels': [0, 0]
        }]
        det_targets = [{
            'boxes': [[10, 10, 50, 50], [100, 100, 150, 150]],
            'labels': [0, 0]
        }]
        r2 = evaluate(det_preds, det_targets)
        assert 'mAP' in r2
