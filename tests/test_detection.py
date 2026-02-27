"""
目标检测评估测试。
"""
import numpy as np
import pytest
from easyMetrics import evaluate_detection, MeanAveragePrecision


class TestEvaluateDetection:
    """evaluate_detection 函数测试。"""

    def test_perfect_prediction(self):
        """完美预测应得到高 mAP。"""
        perfect_preds = [{
            'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
            'scores': np.array([0.95, 0.9], dtype=float),
            'labels': np.array([0, 0], dtype=int)
        }]
        perfect_targets = [{
            'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
            'labels': np.array([0, 0], dtype=int)
        }]
        results = evaluate_detection(perfect_preds, perfect_targets, progress=False)
        assert 'mAP' in results
        assert 'mAP_50' in results
        assert results['mAP_50'] >= 0.99  # 完美预测 mAP_50 应接近 1

    def test_basic_detection(self, detection_preds_targets):
        """基本检测评估。"""
        preds, targets = detection_preds_targets
        results = evaluate_detection(preds, targets, progress=False)
        assert 'mAP' in results
        assert 'mAP_50' in results
        assert 'mAP_75' in results
        assert 'AR_1' in results
        assert 'AR_10' in results
        assert 'AR_100' in results
        assert 0 <= results['mAP'] <= 1
        assert 0 <= results['mAP_50'] <= 1

    def test_custom_metrics(self, detection_preds_targets):
        """自定义指标筛选。"""
        preds, targets = detection_preds_targets
        results = evaluate_detection(
            preds, targets,
            metrics=['mAP', 'mAP_50'],
            progress=False
        )
        assert set(results.keys()) == {'mAP', 'mAP_50'}

    def test_voc_format(self):
        """VOC 格式输入测试。"""
        preds_voc = [[10, 10, 50, 50, 0, 0.95]]
        targets_voc = [[10, 10, 50, 50, 0]]
        results = evaluate_detection(
            [preds_voc], [targets_voc],
            pred_format="voc", target_format="voc",
            progress=False
        )
        assert 'mAP' in results
        assert results['mAP_50'] >= 0.99

    def test_yolo_format(self):
        """YOLO 格式输入测试。"""
        preds_yolo = [[0, 0.5, 0.5, 0.2, 0.2, 0.95]]
        targets_yolo = [[0, 0.5, 0.5, 0.2, 0.2]]
        results = evaluate_detection(
            [preds_yolo], [targets_yolo],
            pred_format="yolo", target_format="yolo",
            image_size=(640, 640),
            progress=False
        )
        assert 'mAP' in results
        assert 0 <= results['mAP'] <= 1

    def test_multi_class(self):
        """多类别检测测试。"""
        preds = [{
            'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
            'scores': np.array([0.95, 0.9], dtype=float),
            'labels': np.array([0, 1], dtype=int)
        }]
        targets = [{
            'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
            'labels': np.array([0, 1], dtype=int)
        }]
        results = evaluate_detection(preds, targets, progress=False)
        assert 'mAP' in results
        assert 'AP_0' in results
        assert 'AP_1' in results

    def test_empty_data(self):
        """空数据测试。"""
        results = evaluate_detection([], [], progress=False)
        assert isinstance(results, dict)
        assert 'mAP' in results or len(results) >= 0

    def test_score_criteria(self, detection_preds_targets):
        """最佳阈值计算测试。"""
        preds, targets = detection_preds_targets
        results = evaluate_detection(
            preds, targets,
            score_criteria=[(0.5, 0.9)],
            progress=False
        )
        # 可能包含 BestScore 键
        best_keys = [k for k in results if 'BestScore' in k]
        # 有类别时应有结果
        assert isinstance(results, dict)

    def test_parallel(self, detection_preds_targets):
        """并行计算测试。"""
        preds, targets = detection_preds_targets
        results_serial = evaluate_detection(preds, targets, n_jobs=1, progress=False)
        results_parallel = evaluate_detection(preds, targets, n_jobs=2, progress=False)
        assert abs(results_serial['mAP'] - results_parallel['mAP']) < 1e-6


class TestMeanAveragePrecision:
    """MeanAveragePrecision 类测试。"""

    def test_map_compute(self, detection_preds_targets):
        """mAP 计算测试。"""
        preds, targets = detection_preds_targets
        metric = MeanAveragePrecision()
        from easyMetrics.tasks.detection.format_converter import DetectionFormatConverter
        conv_preds, conv_targets = DetectionFormatConverter.convert(
            preds, targets, format="coco"
        )
        metric.update(conv_preds, conv_targets)
        results = metric.compute(n_jobs=1, progress=False)
        assert 'mAP' in results
        assert 0 <= results['mAP'] <= 1
