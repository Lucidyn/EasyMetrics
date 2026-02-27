"""
分类评估测试。
"""
import numpy as np
import pytest
from easyMetrics import evaluate_classification, F1Score, AUC, Accuracy


class TestEvaluateClassification:
    """evaluate_classification 函数测试。"""

    def test_binary_classification(self, classification_binary_data):
        """二分类测试。"""
        preds, targets = classification_binary_data
        results = evaluate_classification(preds, targets, average='macro')
        assert 'f1' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'auc' in results
        assert 0 <= results['f1'] <= 1
        assert 0 <= results['auc'] <= 1

    def test_multiclass_classification(self, classification_multiclass_data):
        """多分类测试。"""
        preds, targets = classification_multiclass_data
        results = evaluate_classification(
            preds, targets,
            average='macro', multi_class='ovr'
        )
        assert 'f1' in results
        assert 'auc' in results
        assert 0 <= results['f1'] <= 1
        assert 0 <= results['auc'] <= 1

    def test_multilabel_classification(self, classification_multilabel_data):
        """多标签分类测试。"""
        preds, targets = classification_multilabel_data
        results = evaluate_classification(preds, targets, average='macro')
        assert 'f1' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'auc' in results

    def test_custom_metrics(self, classification_binary_data):
        """自定义指标筛选。"""
        preds, targets = classification_binary_data
        results = evaluate_classification(
            preds, targets,
            metrics=['f1', 'precision']
        )
        assert set(results.keys()) == {'f1', 'precision'}

    def test_average_modes(self, classification_multiclass_data):
        """不同平均方式测试。"""
        preds, targets = classification_multiclass_data
        for avg in ['macro', 'micro', 'weighted']:
            results = evaluate_classification(preds, targets, average=avg)
            assert 'f1' in results
            assert 0 <= results['f1'] <= 1

    def test_perfect_predictions(self):
        """完美预测测试。"""
        preds = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        targets = [0, 1, 0]
        results = evaluate_classification(preds, targets)
        assert results['f1'] == 1.0
        assert results['precision'] == 1.0
        assert results['recall'] == 1.0
        assert results['auc'] == 1.0
        assert results['accuracy'] == 1.0

    def test_accuracy_metric(self):
        """准确率指标测试。"""
        preds = [[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]]
        targets = [0, 1, 0]
        results = evaluate_classification(preds, targets)
        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 1


class TestF1Score:
    """F1Score 指标测试。"""

    def test_f1_binary(self):
        """二分类 F1。"""
        metric = F1Score(average='macro')
        preds = [0, 1, 1, 0]
        targets = [0, 1, 0, 0]
        metric.update(preds, targets)
        results = metric.compute()
        assert 'f1' in results
        assert 0 <= results['f1'] <= 1

    def test_f1_multiclass(self):
        """多分类 F1。"""
        metric = F1Score(average='macro')
        preds = [[0.9, 0.1, 0], [0.1, 0.9, 0], [0.1, 0.1, 0.8]]
        targets = [0, 1, 2]
        metric.update(preds, targets)
        results = metric.compute()
        assert results['f1'] == 1.0

    def test_f1_reset(self):
        """F1 reset 测试。"""
        metric = F1Score(average='macro')
        metric.update([0, 1], [0, 1])
        metric.reset()
        metric.update([1, 0], [1, 0])
        results = metric.compute()
        assert 'f1' in results


class TestAUC:
    """AUC 指标测试。"""

    def test_auc_binary(self):
        """二分类 AUC。"""
        metric = AUC()
        preds = [0.9, 0.3, 0.7, 0.2, 0.8]
        targets = [1, 0, 1, 0, 1]
        metric.update(preds, targets)
        results = metric.compute()
        assert 'auc' in results
        assert 0 <= results['auc'] <= 1

    def test_auc_perfect(self):
        """完美 AUC。"""
        metric = AUC()
        preds = [[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]]
        targets = [1, 0, 1]
        metric.update(preds, targets)
        results = metric.compute()
        assert results['auc'] == 1.0

    def test_auc_multiclass_ovr(self):
        """多分类 AUC (OvR)。"""
        metric = AUC(multi_class='ovr', average='macro')
        preds = [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ]
        targets = [0, 1, 2]
        metric.update(preds, targets)
        results = metric.compute()
        assert results['auc'] == 1.0


class TestAccuracy:
    """Accuracy 指标测试。"""

    def test_accuracy_binary(self):
        """二分类准确率。"""
        metric = Accuracy()
        preds = [0, 1, 1, 0]
        targets = [0, 1, 0, 0]
        metric.update(preds, targets)
        results = metric.compute()
        assert 'accuracy' in results
        assert results['accuracy'] == 0.75  # 3/4 正确 (索引 0,1,3)

    def test_accuracy_perfect(self):
        """完美预测准确率。"""
        metric = Accuracy()
        preds = [[1.0, 0.0], [0.0, 1.0]]
        targets = [0, 1]
        metric.update(preds, targets)
        results = metric.compute()
        assert results['accuracy'] == 1.0
