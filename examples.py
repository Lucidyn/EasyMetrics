"""
EasyMetrics 综合示例文件

此文件包含了 EasyMetrics 库的各种使用示例，包括：
1. 快速入门 - 核心接口的基本用法
2. 详细功能 - 各种高级功能和使用场景
3. 边界情况处理 - 测试空数据等特殊情况
4. 格式支持 - 不同数据格式的使用示例

使用方法：
    python examples.py
"""
import numpy as np
from easyMetrics import evaluate, evaluate_detection, evaluate_classification

def main():
    print("=== EasyMetrics 综合示例 ===")
    print("此示例展示了 EasyMetrics 库的各种功能和使用场景")
    
    # ==================== 快速入门部分 ====================
    print("\n\n=== 第一部分：快速入门 ===")
    print("展示核心接口的基本用法")
    
    # 1. 分类任务示例
    print("\n1. 分类任务示例:")
    # 预测结果和真实标签
    class_preds = [
        [0.9, 0.1],  # 类别 0
        [0.3, 0.7],  # 类别 1
        [0.6, 0.4],  # 类别 0
        [0.8, 0.2]   # 类别 0
    ]
    class_targets = [0, 1, 0, 0]
    
    # 使用统一的 evaluate 函数
    result = evaluate(class_preds, class_targets)
    print(f"自动检测任务类型: {'分类' if 'f1' in result else '检测'}")
    print(f"F1 Score: {result.get('f1', 0):.4f}")
    print(f"AUC: {result.get('auc', 0):.4f}")
    
    # 2. 目标检测示例
    print("\n2. 目标检测示例:")
    # 预测结果和真实标签
    det_preds = [
        {
            'boxes': [[10, 10, 50, 50], [100, 100, 150, 150]],
            'scores': [0.95, 0.9],
            'labels': [0, 0]
        }
    ]
    det_targets = [
        {
            'boxes': [[10, 10, 50, 50], [100, 100, 150, 150]],
            'labels': [0, 0]
        }
    ]
    
    # 使用检测任务专用接口
    det_result = evaluate_detection(det_preds, det_targets)
    print(f"mAP: {det_result.get('mAP', 0):.4f}")
    print(f"mAP_50: {det_result.get('mAP_50', 0):.4f}")
    
    # 3. 使用分类任务专用接口
    print("\n3. 分类任务专用接口:")
    class_result = evaluate_classification(class_preds, class_targets)
    print(f"F1 Score: {class_result.get('f1', 0):.4f}")
    print(f"Precision: {class_result.get('precision', 0):.4f}")
    print(f"Recall: {class_result.get('recall', 0):.4f}")
    print(f"AUC: {class_result.get('auc', 0):.4f}")
    
    # 4. 多分类示例
    print("\n4. 多分类示例:")
    multi_preds = [
        [0.9, 0.05, 0.05],  # 类别 0
        [0.1, 0.8, 0.1],   # 类别 1
        [0.2, 0.3, 0.5],   # 类别 2
        [0.8, 0.1, 0.1]    # 类别 0
    ]
    multi_targets = [0, 1, 2, 0]
    
    # 使用分类任务专用接口
    multi_result = evaluate_classification(multi_preds, multi_targets, multi_class='ovr')
    print(f"多分类 F1 Score: {multi_result.get('f1', 0):.4f}")
    print(f"多分类 AUC: {multi_result.get('auc', 0):.4f}")
    
    # 5. 一行代码完成评估
    print("\n5. 一行代码完成评估:")
    print(f"一行代码评估分类任务: F1={evaluate(class_preds, class_targets).get('f1', 0):.4f}")
    print(f"一行代码评估检测任务: mAP={evaluate(det_preds, det_targets).get('mAP', 0):.4f}")
    
    # ==================== 详细功能部分 ====================
    print("\n\n=== 第二部分：详细功能 ===")
    print("展示各种高级功能和使用场景")
    
    # 准备数据
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

    # 1. 基本用法 - 计算标准 COCO 指标
    print("\n[1] 基本用法 - 计算标准 COCO 指标...")
    results = evaluate_detection(preds, targets)
    
    print(f"mAP (IoU 0.5:0.95): {results['mAP']:.4f}")
    print(f"mAP_50 (IoU 0.5)  : {results['mAP_50']:.4f}")
    print(f"mAP_75 (IoU 0.75) : {results['mAP_75']:.4f}")
    print(f"AR_100 (MaxDets=100): {results['AR_100']:.4f}")

    # 2. 并行计算示例
    print("\n[2] 并行计算示例...")
    print("使用 4 个核心进行评估:")
    results_parallel = evaluate_detection(preds, targets, n_jobs=4)
    print(f"mAP: {results_parallel['mAP']:.4f}")
    print("并行计算测试完成")

    # 3. 自定义指标筛选示例
    print("\n[3] 自定义指标筛选示例...")
    print("只计算 mAP 和 mAP_50 指标:")
    results_custom = evaluate_detection(preds, targets, metrics=['mAP', 'mAP_50'])
    print(f"自定义指标结果: {results_custom}")
    print("自定义指标筛选测试完成")

    # 4. 寻找最佳阈值
    # 场景: 我们希望在 IoU=0.5 的情况下，精度(Precision)至少达到 90%。
    # 问: 我应该设置多高的置信度阈值？
    print("\n[4] 寻找最佳置信度阈值...")
    results_criteria = evaluate_detection(
        preds, targets,
        score_criteria=[(0.5, 0.9)] # (IoU, MinPrecision)
    )
    
    best_thresh = results_criteria.get('BestScore_IoU0.50_P0.90_0')
    print(f"类别 0 在 IoU=0.5 下满足 P>=0.9 的推荐阈值: {best_thresh}")

    # 5. 进度条控制示例
    print("\n[5] 进度条控制示例...")
    print("禁用进度条进行评估:")
    results_no_progress = evaluate_detection(preds, targets, progress=False)
    print(f"mAP: {results_no_progress['mAP']:.4f}")
    print("进度条控制测试完成")

    # 6. 多类别评估示例
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

    # 7. 不同格式输入示例
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

    # ==================== 边界情况处理 ====================
    print("\n\n=== 第三部分：边界情况处理 ===")
    print("测试空数据等特殊情况")
    
    # 1. 测试空数据情况
    print("\n[10] 边界情况处理示例...")
    print("测试空数据情况...")
    empty_preds = []
    empty_targets = []
    try:
        results_empty = evaluate_detection(empty_preds, empty_targets)
        print("空数据测试完成，结果:", results_empty)
    except Exception as e:
        print(f"空数据测试异常: {e}")
    
    # 2. 测试只有预测没有真值的情况
    print("\n测试只有预测没有真值的情况...")
    only_preds = [preds_1]
    only_targets = []
    try:
        results_only_preds = evaluate_detection(only_preds, only_targets)
        print("只有预测没有真值测试完成，结果:", results_only_preds)
    except Exception as e:
        print(f"只有预测没有真值测试异常: {e}")

    # ==================== 分类指标示例 ====================
    print("\n\n=== 第四部分：分类指标示例 ===")
    print("展示分类任务的各种指标和使用场景")
    
    # 1. 二分类示例
    print("\n[11] 分类指标示例 - F1 Score 和 AUC...")
    print("\n二分类示例:")
    # 预测结果和真实标签
    binary_preds = [0.8, 0.3, 0.6, 0.9, 0.2]
    binary_targets = [1, 0, 1, 1, 0]
    
    # 使用 evaluate_classification 函数
    binary_results = evaluate_classification(binary_preds, binary_targets, average='macro')
    print(f"二分类 F1 Score: {binary_results['f1']:.4f}")
    print(f"二分类 Precision: {binary_results['precision']:.4f}")
    print(f"二分类 Recall: {binary_results['recall']:.4f}")
    print(f"二分类 AUC: {binary_results['auc']:.4f}")
    
    # 2. 多分类示例
    print("\n多分类示例:")
    # 多分类预测和标签
    multi_preds = [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]
    multi_targets = [0, 1, 2, 0]
    
    # 使用 evaluate_classification 函数
    multi_results = evaluate_classification(multi_preds, multi_targets, average='macro', multi_class='ovr')
    print(f"多分类 F1 Score: {multi_results['f1']:.4f}")
    print(f"多分类 Precision: {multi_results['precision']:.4f}")
    print(f"多分类 Recall: {multi_results['recall']:.4f}")
    print(f"多分类 AUC: {multi_results['auc']:.4f}")
    
    # 3. 多分类不同方法示例
    print("\n[12] 多分类不同方法示例...")
    # 多分类预测和标签
    multi_class_preds = [
        [0.9, 0.05, 0.05],  # 类别 0
        [0.1, 0.8, 0.1],   # 类别 1
        [0.2, 0.3, 0.5],   # 类别 2
        [0.8, 0.1, 0.1]    # 类别 0
    ]
    multi_class_targets = [0, 1, 2, 0]
    
    # 使用 One-vs-Rest 方法
    results_ovr = evaluate_classification(multi_class_preds, multi_class_targets, multi_class='ovr', average='macro')
    print(f"多分类 AUC (OvR): {results_ovr['auc']:.4f}")
    
    # 4. 多标签示例
    print("\n[13] 多标签示例...")
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
    
    # 使用 evaluate_classification 函数
    multi_label_results = evaluate_classification(multi_label_preds, multi_label_targets, average='macro')
    print(f"多标签 F1 Score: {multi_label_results['f1']:.4f}")
    print(f"多标签 Precision: {multi_label_results['precision']:.4f}")
    print(f"多标签 Recall: {multi_label_results['recall']:.4f}")
    print(f"多标签 AUC: {multi_label_results['auc']:.4f}")
    
    # 5. 统一评估接口示例
    print("\n[14] 统一评估接口示例...")
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
    
    print("\n\n=== 所有示例测试完成！===")
    print("\n总结:")
    print("- evaluate()：统一评测接口，支持自动检测任务类型")
    print("- evaluate_detection()：检测任务专用接口")
    print("- evaluate_classification()：分类任务专用接口")
    print("\n所有接口都支持一行代码完成评估，非常简洁易用！")

if __name__ == "__main__":
    main()
