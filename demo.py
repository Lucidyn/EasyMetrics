"""
EasyMetrics 详细示例文件

此文件包含了 EasyMetrics 库的各种使用示例，包括：
1. 基本用法 - 计算标准 COCO 指标
2. 并行计算示例 - 提高大规模数据集的评估速度
3. 自定义指标筛选示例 - 只计算关心的指标
4. 寻找最佳置信度阈值示例 - 自动计算满足特定精度要求的阈值
5. 进度条控制示例 - 控制是否显示评估进度
6. 多类别评估示例 - 评估包含多个类别的检测结果
7. 不同格式输入示例 - 测试 VOC 格式输入
8. YOLO 格式输入示例 - 测试 YOLO 格式输入
9. 混合格式输入示例 - 测试不同格式的混合使用
10. 边界情况处理示例 - 测试空数据等边界情况
11. 多类别多尺度目标示例 - 测试不同尺度目标的检测性能
12. 批量 YOLO 格式数据示例 - 测试多张图片的 YOLO 格式数据

使用方法：
    python demo.py
"""
import numpy as np
from easyMetrics.tasks.detection import evaluate_detection
from easyMetrics import evaluate

def main():
    print("=== EasyMetrics 详细示例 ===")
    print("此示例展示了 EasyMetrics 库的各种功能和使用场景")

    # 1. 准备数据
    # 格式: [x1, y1, x2, y2] - 左上角和右下角坐标
    # 假设有 2 张图片，每张图片对应一个字典
    
    # 图片 1: 预测完全正确
    # 数据结构说明:
    # - boxes: 边界框坐标，形状为 (N, 4)，N 为目标数量
    # - scores: 置信度分数，形状为 (N,)
    # - labels: 类别索引，形状为 (N,)
    preds_1 = {
        'boxes': np.array([[10, 10, 50, 50]], dtype=float),
        'scores': np.array([0.95], dtype=float),
        'labels': np.array([0], dtype=int)
    }
    targets_1 = {
        'boxes': np.array([[10, 10, 50, 50]], dtype=float),
        'labels': np.array([0], dtype=int)  # 真值不需要 scores
    }

    # 图片 2: 有一个误检 (FP) 和一个漏检 (FN)
    # 预测: 
    # - [100, 100, 150, 150] (IoU=0.9, 正确)
    # - [0, 0, 20, 20] (误检)
    # 真值:
    # - [100, 100, 150, 150] (被正确检测)
    # - [200, 200, 250, 250] (未被检测到，漏检)
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
            [200, 200, 250, 250] # 这个没被检测到 (漏检)
        ], dtype=float),
        'labels': np.array([0, 0], dtype=int)
    }

    # 组合成完整的数据集
    # preds: 所有图片的预测结果列表
    # targets: 所有图片的真实标签列表
    preds = [preds_1, preds_2]
    targets = [targets_1, targets_2]
    
    print(f"数据集准备完成: {len(preds)} 张图片, {sum(len(p['boxes']) for p in preds)} 个预测, {sum(len(t['boxes']) for t in targets)} 个真值")

    # 2. 基本用法 - 计算标准 COCO 指标
    print("\n[1] 基本用法 - 计算标准 COCO 指标...")
    results = evaluate_detection(preds, targets)
    
    print(f"mAP (IoU 0.5:0.95): {results['mAP']:.4f}")
    print(f"mAP_50 (IoU 0.5)  : {results['mAP_50']:.4f}")
    print(f"mAP_75 (IoU 0.75) : {results['mAP_75']:.4f}")
    print(f"AR_100 (MaxDets=100): {results['AR_100']:.4f}")

    # 3. 并行计算示例
    print("\n[2] 并行计算示例...")
    print("使用 4 个核心进行评估:")
    results_parallel = evaluate_detection(preds, targets, n_jobs=4)
    print(f"mAP: {results_parallel['mAP']:.4f}")
    print("并行计算测试完成")

    # 4. 自定义指标筛选示例
    print("\n[3] 自定义指标筛选示例...")
    print("只计算 mAP 和 mAP_50 指标:")
    results_custom = evaluate_detection(preds, targets, metrics=['mAP', 'mAP_50'])
    print(f"自定义指标结果: {results_custom}")
    print("自定义指标筛选测试完成")

    # 5. 寻找最佳阈值
    # 场景: 我们希望在 IoU=0.5 的情况下，精度(Precision)至少达到 90%。
    # 问: 我应该设置多高的置信度阈值？
    print("\n[4] 寻找最佳置信度阈值...")
    results_criteria = evaluate_detection(
        preds, targets,
        score_criteria=[(0.5, 0.9)] # (IoU, MinPrecision)
    )
    
    best_thresh = results_criteria.get('BestScore_IoU0.50_P0.90_0')
    print(f"类别 0 在 IoU=0.5 下满足 P>=0.9 的推荐阈值: {best_thresh}")

    # 6. 进度条控制示例
    print("\n[5] 进度条控制示例...")
    print("禁用进度条进行评估:")
    results_no_progress = evaluate_detection(preds, targets, progress=False)
    print(f"mAP: {results_no_progress['mAP']:.4f}")
    print("进度条控制测试完成")

    # 7. 多类别评估示例
    print("\n[6] 多类别评估示例...")
    # 准备多类别数据
    preds_multi = [{
        'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
        'scores': np.array([0.95, 0.9], dtype=float),
        'labels': np.array([0, 1], dtype=int)  # 两个不同的类别
    }]
    targets_multi = [{
        'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
        'labels': np.array([0, 1], dtype=int)  # 两个不同的类别
    }]
    
    results_multi = evaluate_detection(preds_multi, targets_multi)
    print(f"多类别评估 mAP: {results_multi['mAP']:.4f}")
    print(f"类别 0 的 AP: {results_multi['AP_0']:.4f}")
    print(f"类别 1 的 AP: {results_multi['AP_1']:.4f}")
    print("多类别评估测试完成")

    # 8. 不同格式输入示例
    print("\n[7] 不同格式输入示例...")
    # VOC 格式预测
    preds_voc = [[10, 10, 50, 50, 0, 0.95]]  # [x1, y1, x2, y2, class_id, confidence]
    # VOC 格式真值
    targets_voc = [[10, 10, 50, 50, 0]]  # [x1, y1, x2, y2, class_id]
    
    results_voc = evaluate_detection(
        [preds_voc], [targets_voc], 
        pred_format="voc", 
        target_format="voc"
    )
    print(f"VOC 格式输入评估 mAP: {results_voc['mAP']:.4f}")
    
    # YOLO 格式输入示例
    print("\n[8] YOLO 格式输入示例...")
    # YOLO 格式: [class_id, x_center, y_center, width, height, confidence]
    # 注意: 坐标是归一化的 (0-1)
    preds_yolo = [[0, 0.5, 0.5, 0.2, 0.2, 0.95]]
    targets_yolo = [[0, 0.5, 0.5, 0.2, 0.2]]
    
    results_yolo = evaluate_detection(
        [preds_yolo], [targets_yolo],
        pred_format="yolo",
        target_format="yolo",
        image_size=(640, 640)  # YOLO 格式需要的图像尺寸
    )
    print(f"YOLO 格式输入评估 mAP: {results_yolo['mAP']:.4f}")
    
    # 混合格式输入示例
    print("\n[9] 混合格式输入示例...")
    # 预测值使用 YOLO 格式，真值使用 VOC 格式
    results_mixed = evaluate_detection(
        [preds_yolo], [targets_voc],
        pred_format="yolo",
        target_format="voc",
        image_size=(640, 640)
    )
    print(f"混合格式输入评估 mAP: {results_mixed['mAP']:.4f}")
    print("不同格式输入测试完成")

    # 10. 边界情况处理示例
    print("\n[10] 边界情况处理示例...")
    # 测试空数据情况
    print("测试空数据情况...")
    empty_preds = []
    empty_targets = []
    try:
        results_empty = evaluate_detection(empty_preds, empty_targets)
        print("空数据测试完成，结果:", results_empty)
    except Exception as e:
        print(f"空数据测试异常: {e}")
    
    # 测试只有预测没有真值的情况
    print("\n测试只有预测没有真值的情况...")
    only_preds = [preds_1]
    only_targets = []
    try:
        results_only_preds = evaluate_detection(only_preds, only_targets)
        print("只有预测没有真值测试完成，结果:", results_only_preds)
    except Exception as e:
        print(f"只有预测没有真值测试异常: {e}")
    
    # 11. 多类别多尺度目标示例
    print("\n[11] 多类别多尺度目标示例...")
    # 准备多类别多尺度数据
    # 类别 0: 大目标
    # 类别 1: 中目标
    # 类别 2: 小目标
    complex_preds = [{
        'boxes': np.array([
            [50, 50, 200, 200],   # 大目标
            [250, 250, 300, 300], # 中目标
            [350, 350, 360, 360]  # 小目标
        ], dtype=float),
        'scores': np.array([0.95, 0.9, 0.85], dtype=float),
        'labels': np.array([0, 1, 2], dtype=int)
    }]
    complex_targets = [{
        'boxes': np.array([
            [50, 50, 200, 200],   # 大目标
            [250, 250, 300, 300], # 中目标
            [350, 350, 360, 360], # 小目标
            [400, 400, 410, 410]  # 小目标 (漏检)
        ], dtype=float),
        'labels': np.array([0, 1, 2, 2], dtype=int)
    }]
    
    results_complex = evaluate_detection(
        complex_preds, complex_targets,
        metrics=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    )
    print(f"多类别多尺度评估 mAP: {results_complex['mAP']:.4f}")
    print(f"mAP_50: {results_complex['mAP_50']:.4f}")
    print(f"mAP_75: {results_complex['mAP_75']:.4f}")
    print(f"小目标 mAP_s: {results_complex['mAP_s']:.4f}")
    print(f"中目标 mAP_m: {results_complex['mAP_m']:.4f}")
    print(f"大目标 mAP_l: {results_complex['mAP_l']:.4f}")
    
    # 12. 批量 YOLO 格式数据示例
    print("\n[12] 批量 YOLO 格式数据示例...")
    # 准备批量 YOLO 格式数据
    # 多张图片的 YOLO 格式预测和真值
    
    # 图片 1: 包含两个目标 (行人 + 车辆)
    # YOLO 格式: [class_id, x_center, y_center, width, height, confidence]
    preds_yolo_batch_1 = [
        [0, 0.3, 0.4, 0.2, 0.3, 0.95],  # 行人
        [1, 0.7, 0.6, 0.3, 0.4, 0.9]     # 车辆
    ]
    targets_yolo_batch_1 = [
        [0, 0.3, 0.4, 0.2, 0.3],  # 行人
        [1, 0.7, 0.6, 0.3, 0.4]     # 车辆
    ]
    
    # 图片 2: 包含一个目标 (行人)
    preds_yolo_batch_2 = [
        [0, 0.5, 0.5, 0.2, 0.2, 0.85]  # 行人
    ]
    targets_yolo_batch_2 = [
        [0, 0.5, 0.5, 0.2, 0.2],  # 行人
        [1, 0.8, 0.3, 0.2, 0.2]     # 车辆 (漏检)
    ]
    
    # 图片 3: 包含三个目标 (两个行人 + 一个车辆)
    preds_yolo_batch_3 = [
        [0, 0.2, 0.3, 0.15, 0.25, 0.92],  # 行人 1
        [0, 0.8, 0.7, 0.18, 0.22, 0.88],  # 行人 2
        [1, 0.5, 0.6, 0.25, 0.3, 0.9]      # 车辆
    ]
    targets_yolo_batch_3 = [
        [0, 0.2, 0.3, 0.15, 0.25],  # 行人 1
        [0, 0.8, 0.7, 0.18, 0.22],  # 行人 2
        [1, 0.5, 0.6, 0.25, 0.3]      # 车辆
    ]
    
    # 组合成批量数据
    preds_yolo_batch = [preds_yolo_batch_1, preds_yolo_batch_2, preds_yolo_batch_3]
    targets_yolo_batch = [targets_yolo_batch_1, targets_yolo_batch_2, targets_yolo_batch_3]
    
    print(f"批量 YOLO 数据准备完成: {len(preds_yolo_batch)} 张图片")
    
    # 评估批量 YOLO 格式数据
    results_yolo_batch = evaluate_detection(
        preds_yolo_batch, targets_yolo_batch,
        pred_format="yolo",
        target_format="yolo",
        image_size=(640, 640),  # YOLO 格式需要的图像尺寸
        n_jobs=2  # 使用并行计算加速
    )
    
    print(f"批量 YOLO 格式评估 mAP: {results_yolo_batch['mAP']:.4f}")
    print(f"批量 YOLO 格式评估 mAP_50: {results_yolo_batch['mAP_50']:.4f}")
    print(f"批量 YOLO 格式评估 mAP_75: {results_yolo_batch['mAP_75']:.4f}")
    print(f"批量 YOLO 格式评估 AR_100: {results_yolo_batch['AR_100']:.4f}")
    
    # 按类别查看结果
    if 'AP_0' in results_yolo_batch:
        print(f"行人类别 (0) AP: {results_yolo_batch['AP_0']:.4f}")
    if 'AP_1' in results_yolo_batch:
        print(f"车辆类别 (1) AP: {results_yolo_batch['AP_1']:.4f}")
    
    print("批量 YOLO 格式数据测试完成")
    
    # 13. 分类指标示例 - F1 Score
    print("\n[13] 分类指标示例 - F1 Score...")
    from easyMetrics import F1Score
    
    # 二分类示例
    print("\n二分类示例:")
    f1_metric = F1Score(average='macro')
    
    # 预测结果和真实标签
    binary_preds = [0.8, 0.3, 0.6, 0.9, 0.2]
    binary_targets = [1, 0, 1, 1, 0]
    
    f1_metric.update(binary_preds, binary_targets)
    binary_results = f1_metric.compute()
    print(f"二分类 F1 Score: {binary_results['f1']:.4f}")
    print(f"二分类 Precision: {binary_results['precision']:.4f}")
    print(f"二分类 Recall: {binary_results['recall']:.4f}")
    
    # 多分类示例
    print("\n多分类示例:")
    f1_metric_multi = F1Score(average='macro', num_classes=3)
    
    # 多分类预测和标签
    multi_preds = [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]
    multi_targets = [0, 1, 2, 0]
    
    f1_metric_multi.update(multi_preds, multi_targets)
    multi_results = f1_metric_multi.compute()
    print(f"多分类 F1 Score: {multi_results['f1']:.4f}")
    print(f"多分类 Precision: {multi_results['precision']:.4f}")
    print(f"多分类 Recall: {multi_results['recall']:.4f}")
    
    # 14. 分类指标示例 - AUC
    print("\n[14] 分类指标示例 - AUC...")
    from easyMetrics import AUC
    
    auc_metric = AUC(method='trapezoidal')
    
    # AUC 计算需要概率值
    auc_preds = [0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.5]
    auc_targets = [1, 0, 1, 1, 0, 1, 0, 1]
    
    auc_metric.update(auc_preds, auc_targets)
    auc_results = auc_metric.compute()
    print(f"AUC 值: {auc_results['auc']:.4f}")
    
    # 测试不同的计算方法
    auc_metric_linear = AUC(method='linear')
    auc_metric_linear.update(auc_preds, auc_targets)
    auc_results_linear = auc_metric_linear.compute()
    print(f"AUC 值 (线性插值): {auc_results_linear['auc']:.4f}")
    
    # 15. 多分类示例 - AUC
    print("\n[15] 多分类示例 - AUC...")
    # 多分类预测和标签
    multi_class_preds = [
        [0.9, 0.05, 0.05],  # 类别 0
        [0.1, 0.8, 0.1],   # 类别 1
        [0.2, 0.3, 0.5],   # 类别 2
        [0.8, 0.1, 0.1]    # 类别 0
    ]
    multi_class_targets = [0, 1, 2, 0]
    
    # 使用 One-vs-Rest 方法
    auc_metric_multi = AUC(method='trapezoidal', multi_class='ovr', average='macro')
    auc_metric_multi.update(multi_class_preds, multi_class_targets)
    multi_auc_results = auc_metric_multi.compute()
    print(f"多分类 AUC (OvR): {multi_auc_results['auc']:.4f}")
    
    # 使用 One-vs-One 方法
    auc_metric_multi_ovo = AUC(method='trapezoidal', multi_class='ovo', average='macro')
    auc_metric_multi_ovo.update(multi_class_preds, multi_class_targets)
    multi_auc_results_ovo = auc_metric_multi_ovo.compute()
    print(f"多分类 AUC (OvO): {multi_auc_results_ovo['auc']:.4f}")
    
    # 16. 多标签示例
    print("\n[16] 多标签示例...")
    # 多标签预测和标签
    multi_label_preds = [
        [0.9, 0.8, 0.1],  # 标签 0 和 1
        [0.2, 0.6, 0.7],   # 标签 1 和 2
        [0.8, 0.3, 0.2],   # 标签 0
        [0.1, 0.2, 0.3]    # 无标签
    ]
    multi_label_targets = [
        [1, 1, 0],  # 标签 0 和 1
        [0, 1, 1],   # 标签 1 和 2
        [1, 0, 0],   # 标签 0
        [0, 0, 0]    # 无标签
    ]
    
    # F1 Score 多标签示例
    print("\n多标签 F1 Score:")
    f1_metric_multi_label = F1Score(average='macro')
    f1_metric_multi_label.update(multi_label_preds, multi_label_targets)
    multi_label_f1_results = f1_metric_multi_label.compute()
    print(f"多标签 F1 Score: {multi_label_f1_results['f1']:.4f}")
    print(f"多标签 Precision: {multi_label_f1_results['precision']:.4f}")
    print(f"多标签 Recall: {multi_label_f1_results['recall']:.4f}")
    
    # AUC 多标签示例
    print("\n多标签 AUC:")
    auc_metric_multi_label = AUC(method='trapezoidal', average='macro')
    auc_metric_multi_label.update(multi_label_preds, multi_label_targets)
    multi_label_auc_results = auc_metric_multi_label.compute()
    print(f"多标签 AUC: {multi_label_auc_results['auc']:.4f}")
    
    # 17. 统一评估接口示例
    print("\n[17] 统一评估接口示例...")
    # 分类任务评估
    print("\n分类任务评估:")
    unified_results = evaluate(
        multi_class_preds, 
        multi_class_targets, 
        task='classification'
    )
    print(f"统一接口分类评估结果: F1={unified_results['f1']:.4f}, AUC={unified_results['auc']:.4f}")
    
    # 检测任务评估（使用之前的数据）
    print("\n检测任务评估:")
    unified_detection_results = evaluate(
        preds, 
        targets, 
        task='detection'
    )
    print(f"统一接口检测评估结果: mAP={unified_detection_results['mAP']:.4f}")
    
    print("\n=== 所有示例测试完成！===")

if __name__ == "__main__":
    main()
